import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt



# --- Random seed -------------------------------------------------------
SEED = 42 #10,42, 40, 41
DATA_SEED = SEED

# --- Grid / domain -----------------------------------------------------
DX = 50.0          # cell size in x (m)
DY = 50.0          # cell size in y (m)
DZ = 50.0          # cell size in z (m)
X_MAX = 1000.0     # domain extent in x (m)
Y_MAX = 1000.0     # domain extent in y (m)
Z_MAX = 500.0      # domain extent in z (m)

# --- Block model -------------------------------------------------------
RHO_BG  = 0.0      # background density contrast (kg/m³)
RHO_BLK = 400.0    # block density contrast (kg/m³)

# --- Noise level -------------------------------------------------------
NOISE_LEVEL = 0.01  # fraction of gz_true std

# --- Training / optimisation -------------------------------------------
GAMMA   = 1.0       # data-term weight
EPOCHS  = 500       # number of training epochs
LR      = 1e-2      # Adam learning rate

# --- Early stopping ----------------------------------------------------
# Stop near the expected noise floor to reduce overfitting to 1% noise.
USE_EARLY_STOPPING = True
EARLY_STOP_MIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
EARLY_STOP_TARGET = 1.0
EARLY_STOP_TOL = 0.05
EARLY_STOP_OVERFIT_PATIENCE = 5

# --- INR network -------------------------------------------------------
HIDDEN       = 256   # hidden-layer width
DEPTH        = 4     # number of hidden layers
RHO_ABS_MAX  = 600.0 # tanh output scaling (kg/m³)

# --- Encoding strategy -------------------------------------------------
# Compare only the two encoded models in this script.
ENCODING_ORDER = ('positional', 'hash')

ENCODING_CONFIGS = {
    'positional': dict(num_freqs=2),
    'hash':       dict(n_levels=2, n_features_per_level=2,
                       log2_hashmap_size=17, base_resolution=4,
                       finest_resolution=128),
}

# --- Plotting -----------------------------------------------------------
CMAP          = 'Spectral_r'   # colormap for all plots
INV_VMAX      = 250       # fixed colorbar max for inverted model
FIG_DPI       = 300
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 10



class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (NeRF-style Fourier features).

    Intuition
    ---------
    Plain MLPs are biased toward learning low-frequency functions — they
    tend to produce overly smooth outputs and struggle to represent sharp
    edges or fine detail ("spectral bias").  Positional encoding fights
    this by lifting each input coordinate into a set of sine and cosine
    waves at *exponentially increasing* frequencies:

        γ(p) = [sin(2⁰πp), cos(2⁰πp), sin(2¹πp), cos(2¹πp), …]

    Think of it as giving the network a set of "rulers" at different
    scales: the lowest frequency captures the overall trend, while
    higher frequencies let the network resolve increasingly fine spatial
    variation.  The resulting feature vector is fixed (no learnable
    parameters in the encoding itself).

    Trade-offs
    ----------
    • Simple and deterministic — no extra learnable parameters.
    • The frequency ladder is fixed (powers of 2), so the spectrum can
      have gaps and may not cover all scales equally well.
    • Too few frequencies → overly smooth model; too many → potential
      noise fitting and slower convergence.
    • Works best with a sufficiently deep/wide MLP.

    Args:
        num_freqs:      Number of frequency octaves (L).  Output grows
                        as input_dim × (1 + 2L) with include_input.
        include_input:  Whether to prepend the raw (x, y, z) values.
        input_dim:      Spatial dimensionality (default 3).
    """
    def __init__(self, num_freqs=8, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        self.out_dim = input_dim * (1 + 2 * num_freqs) if include_input else input_dim * 2 * num_freqs

    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)


class HashEncoding(nn.Module):
    """Multi-resolution hash encoding (Müller et al., 2022 / Instant-NGP).

    Intuition
    ---------
    Imagine overlaying your 3-D domain with a stack of voxel grids, from
    very coarse (e.g. 4³) to very fine (e.g. 128³).  At each resolution
    level, every voxel stores a small learnable feature vector.  To
    encode a point, you look up its 8 surrounding voxel corners at every
    level, trilinearly interpolate, and concatenate across levels.

    The "hash" trick makes this memory-efficient: instead of allocating a
    full 3-D grid (which grows as O(R³)), each level maps voxel corners
    to a fixed-size hash table.  Collisions (two corners sharing one
    slot) are resolved implicitly by the gradient-based optimisation —
    the network learns to disentangle them.

    The result is an encoding that is:
    • **Adaptive**: features are *learned*, so detail concentrates where
      the data demands it (unlike fixed Fourier features).
    • **Multi-scale**: coarse levels capture large-scale trends; fine
      levels capture sharp boundaries and small anomalies.
    • **Memory-bounded**: hash table size is constant regardless of
      how fine the resolution is — crucial for large 3-D domains.

    Trade-offs
    ----------
    • Most flexible encoding — can represent both smooth and sharp models.
    • Hash collisions at fine levels can introduce small artifacts or a
      noise floor if the hash table is too small (increase
      log2_hashmap_size to mitigate).
    • Many hyperparameters to tune (n_levels, base/finest resolution,
      hash table size).
    • Learnable parameters mean more total parameters to optimise, and
      they may overfit noisy data without appropriate regularisation.

    Args:
        n_levels:               Number of resolution levels.  More levels
                                give smoother multi-scale interpolation.
        n_features_per_level:   Feature dimensions stored per hash entry.
                                Typically 2; larger values add capacity.
        log2_hashmap_size:      Log₂ of hash-table size per level.
                                2^19 ≈ 500 k entries is a common default.
        base_resolution:        Coarsest voxel-grid resolution (e.g. 4).
        finest_resolution:      Finest voxel-grid resolution (e.g. 128).
                                Intermediate levels are spaced
                                geometrically between base and finest.
        input_dim:              Spatial dimensionality (default 3).
    """
    def __init__(self, n_levels=16, n_features_per_level=2,
                 log2_hashmap_size=19, base_resolution=16,
                 finest_resolution=512, input_dim=3):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.input_dim = input_dim
        self.out_dim = n_levels * n_features_per_level
        self.hashmap_size = 2 ** log2_hashmap_size

        if n_levels > 1:
            self.growth_factor = np.exp(
                (np.log(finest_resolution) - np.log(base_resolution))
                / (n_levels - 1))
        else:
            self.growth_factor = 1.0
        self.base_resolution = base_resolution

        # Learnable hash tables (one per level)
        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for tbl in self.hash_tables:
            nn.init.uniform_(tbl.weight, -1e-4, 1e-4)

        # Large primes for the spatial hash
        self.register_buffer(
            'primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))

    def _hash(self, coords_int):
        """Spatial hash: integer grid coords -> hash-table index."""
        result = torch.zeros(coords_int.shape[:-1],
                             dtype=torch.long, device=coords_int.device)
        for d in range(self.input_dim):
            result ^= coords_int[..., d] * self.primes[d]
        return result % self.hashmap_size

    def forward(self, x):
        # Normalise to [0, 1] using per-batch bounds
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        x_scaled = (x - x_min) / (x_max - x_min + 1e-8)

        outputs = []
        for level in range(self.n_levels):
            resolution = int(self.base_resolution * (self.growth_factor ** level))
            x_grid = x_scaled * resolution            # (N, 3)
            x_floor = torch.floor(x_grid).long()      # voxel origin
            x_frac  = x_grid - x_floor.float()        # interpolation weight

            # Eight voxel corners
            corners = []
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        corners.append(
                            x_floor + torch.tensor([dx, dy, dz], device=x.device))
            corners = torch.stack(corners, dim=1)      # (N, 8, 3)

            indices  = self._hash(corners)             # (N, 8)
            features = self.hash_tables[level](indices) # (N, 8, F)

            # Trilinear interpolation weights
            wx, wy, wz = (x_frac[:, 0:1],
                          x_frac[:, 1:2],
                          x_frac[:, 2:3])
            weights = torch.stack([
                (1-wx)*(1-wy)*(1-wz),  wx*(1-wy)*(1-wz),
                (1-wx)*   wy *(1-wz),  wx*   wy *(1-wz),
                (1-wx)*(1-wy)*   wz ,  wx*(1-wy)*   wz ,
                (1-wx)*   wy *   wz ,  wx*   wy *   wz ,
            ], dim=1)                                  # (N, 8, 1)

            outputs.append((weights * features).sum(dim=1))  # (N, F)

        return torch.cat(outputs, dim=-1)              # (N, n_levels*F)


def create_encoding(encoding_type, **kwargs):
    """Factory: build an encoding module by name.

    Supported types
    ---------------
    positional  – Sinusoidal Fourier features (Mildenhall et al., 2020
                  / NeRF).  Fixed frequencies on a power-of-2 ladder.
                  Simple, no learnable params; good general-purpose
                  baseline.  Tune `num_freqs` for resolution.

    hash        – Multi-resolution hash tables (Müller et al., 2022 /
                  Instant-NGP).  Learnable feature grids at multiple
                  resolutions compressed via spatial hashing.  Most
                  flexible; best for sharp boundaries.  More params &
                  hyperparameters.
    """
    if encoding_type == 'positional':
        enc = PositionalEncoding(
            num_freqs=kwargs.get('num_freqs', 8),
            include_input=kwargs.get('include_input', True))
        return enc
    if encoding_type == 'hash':
        return HashEncoding(
            n_levels=kwargs.get('n_levels', 16),
            n_features_per_level=kwargs.get('n_features_per_level', 2),
            log2_hashmap_size=kwargs.get('log2_hashmap_size', 19),
            base_resolution=kwargs.get('base_resolution', 16),
            finest_resolution=kwargs.get('finest_resolution', 512))
    raise ValueError(f"Unknown encoding type: {encoding_type}. Expected 'positional' or 'hash'.")


class DensityContrastINR(nn.Module):
    """INR density-contrast model with pluggable spatial encoding."""
    def __init__(self, encoding_type='positional', hidden=256, depth=5,
                 rho_abs_max=600.0, **encoding_kwargs):
        super().__init__()
        self.pe = create_encoding(encoding_type, **encoding_kwargs)
        in_dim = self.pe.out_dim
        layers = []
        h = hidden
        layers += [nn.Linear(in_dim, h), nn.LeakyReLU(0.01)]
        for _ in range(depth - 1):
            layers += [nn.Linear(h, h), nn.LeakyReLU(0.01)]
        layers += [nn.Linear(h, 1)]
        self.net = nn.Sequential(*layers)
        self.rho_abs_max = float(rho_abs_max)

    def forward(self, x):
        z = self.pe(x)
        out = self.net(z)
        return self.rho_abs_max * torch.tanh(out)

# ──────────────────────────────────────────────────────────────────────
#                     UTILITY FUNCTIONS
# ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"Seed = {seed}")


def capture_rng_state():
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if torch.cuda.is_available() and 'cuda' in state:
        torch.cuda.set_rng_state_all(state['cuda'])

def A_integral_torch(x, y, z):
    eps = 1e-20
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + r) + eps) +
             y * torch.log(torch.abs(x + r) + eps) -
             z * torch.atan2(x * y, z * r + eps))

@torch.inference_mode()
def construct_sensitivity_matrix_G_torch(cell_grid, data_points, d1, d2, device):
    Gamma = 6.67430e-11
    cx = cell_grid[:, 0].unsqueeze(0)
    cy = cell_grid[:, 1].unsqueeze(0)
    cz = cell_grid[:, 2].unsqueeze(0)
    czh = cell_grid[:, 3].unsqueeze(0)
    ox = data_points[:, 0].unsqueeze(1)
    oy = data_points[:, 1].unsqueeze(1)
    oz = data_points[:, 2].unsqueeze(1)
    x2, x1 = (cx + d1/2) - ox, (cx - d1/2) - ox
    y2, y1 = (cy + d2/2) - oy, (cy - d2/2) - oy
    z2, z1 = (cz + czh) - oz, (cz - czh) - oz
    A = (A_integral_torch(x2, y2, z2) - A_integral_torch(x2, y2, z1) -
         A_integral_torch(x2, y1, z2) + A_integral_torch(x2, y1, z1) -
         A_integral_torch(x1, y2, z2) + A_integral_torch(x1, y2, z1) +
         A_integral_torch(x1, y1, z2) - A_integral_torch(x1, y1, z1))
    return (Gamma * A).to(device)

def generate_grf_torch(nx, ny, nz, dx, dy, dz, lam, nu, sigma, device):
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * torch.pi
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = Kx**2 + Ky**2 + Kz**2
    P = (k2 + (1/lam**2))**(-nu - 1.5)
    P[0, 0, 0] = 0
    noise = torch.randn(nx, ny, nz, dtype=torch.complex64, device=device)
    f = noise * torch.sqrt(P)
    m = torch.real(torch.fft.ifftn(f))
    m = (m - m.mean()) / (m.std() + 1e-9)
    return sigma * m

def train_inr(model, opt, coords_norm, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, cfg):
    history = {"total": [], "gravity": []}
    use_early_stopping = cfg.get('use_early_stopping', False)
    min_epochs = cfg.get('early_stop_min_epochs', 0)
    patience = cfg.get('early_stop_patience', 0)
    target = cfg.get('early_stop_target', 1.0)
    tol = cfg.get('early_stop_tol', 0.0)
    overfit_patience = cfg.get('early_stop_overfit_patience', 0)
    best_gap = float('inf')
    best_epoch = -1
    best_weighted_mse = None
    best_state = None
    in_band_count = 0
    overfit_count = 0

    for ep in range(cfg['epochs']):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = torch.matmul(G, m_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        data_term = cfg['gamma'] * torch.mean((Wd * residual) ** 2)
        loss = data_term
        loss.backward()
        opt.step()
        history['gravity'].append(float(data_term.item()))
        history['total'].append(float(loss.item()))

        if use_early_stopping:
            weighted_mse = float(data_term.item() / cfg['gamma'])
            gap = abs(weighted_mse - target)
            if gap < best_gap:
                best_gap = gap
                best_epoch = ep
                best_weighted_mse = weighted_mse
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if ep + 1 >= min_epochs:
                if target - tol <= weighted_mse <= target + tol:
                    in_band_count += 1
                else:
                    in_band_count = 0

                if weighted_mse < target - tol:
                    overfit_count += 1
                else:
                    overfit_count = 0

                if in_band_count >= patience:
                    print(
                        f"Early stopping at epoch {ep:4d} | "
                        f"reason = near noise floor | "
                        f"loss = {weighted_mse:.3f} | "
                        f"best epoch = {best_epoch}"
                    )
                    break

                if overfit_count >= overfit_patience:
                    print(
                        f"Early stopping at epoch {ep:4d} | "
                        f"reason = below noise floor | "
                        f"loss = {weighted_mse:.3f} | "
                        f"best epoch = {best_epoch}"
                    )
                    break

        if ep % 50 == 0 or ep == cfg['epochs'] - 1:
            print(f"Epoch {ep:4d} | loss {history['gravity'][-1]:.3e}")

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        history['best_epoch'] = best_epoch
        history['best_weighted_mse'] = best_weighted_mse
    else:
        history['best_epoch'] = len(history['gravity']) - 1
        history['best_weighted_mse'] = history['gravity'][-1] / cfg['gamma']

    return history

def make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0):
    m = torch.full((Nx, Ny, Nz), rho_bg)
    for i in range(7):
        z_idx = 1 + i
        y_start, y_end = 11 - i, 16 - i
        x_start, x_end = 7, 13
        if 0 <= z_idx < Nz:
            ys, ye = max(0, y_start), min(Ny, y_end)
            xs, xe = max(0, x_start), min(Nx, x_end)
            m[xs:xe, ys:ye, z_idx] = rho_blk
    return m.view(-1), m

def get_block_boundaries(Nx, Ny, Nz):
    boundaries = []
    for i in range(7):
        z_idx = 1 + i
        y_start, y_end = 11 - i, 16 - i
        x_start, x_end = 7, 13
        if 0 <= z_idx < Nz:
            ys, ye = max(0, y_start), min(Ny, y_end)
            xs, xe = max(0, x_start), min(Nx, x_end)
            boundaries.append((xs, xe, ys, ye, z_idx))
    return boundaries


def add_xy_outline(ax, boundaries, x, y, dx, dy, iz):
    boundary_for_z = next((b for b in boundaries if b[4] == iz), None)
    if boundary_for_z:
        xs, xe, ys, ye, _ = boundary_for_z
        rect = plt.Rectangle((x[xs] - dx / 2, y[ys] - dy / 2),
                             (xe - xs) * dx,
                             (ye - ys) * dy,
                             edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(rect)


def add_xz_outline(ax, boundaries, x, z, dx, dz, iy):
    z_indices_in_slice = []
    x_range = None
    for xs, xe, ys, ye, z_idx in boundaries:
        if ys <= iy < ye:
            z_indices_in_slice.append(z_idx)
            if x_range is None:
                x_range = (xs, xe)
    if z_indices_in_slice and x_range:
        min_z_idx, max_z_idx = min(z_indices_in_slice), max(z_indices_in_slice)
        xs, xe = x_range
        rect = plt.Rectangle((x[xs] - dx / 2, z[min_z_idx] - dz / 2),
                             (xe - xs) * dx,
                             (max_z_idx - min_z_idx + 1) * dz,
                             edgecolor='white', facecolor='none', linewidth=2)
        ax.add_patch(rect)


def add_yz_outline(ax, boundaries, y, z, dy, dz, ix):
    for xs, xe, ys, ye, z_idx in boundaries:
        if xs <= ix < xe:
            rect = plt.Rectangle((y[ys] - dy / 2, z[z_idx] - dz / 2),
                                 (ye - ys) * dy,
                                 dz,
                                 edgecolor='white', facecolor='none', linewidth=2)
            ax.add_patch(rect)


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def encoding_label(name):
    return 'Positional Encoding' if name == 'positional' else 'Hash Encoding'

def run():
    set_seed(DATA_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    dx, dy, dz = DX, DY, DZ
    x = np.arange(0.0, X_MAX + dx, dx)
    y = np.arange(0.0, Y_MAX + dy, dy)
    z = np.arange(0.0, Z_MAX + dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    Xc = x.astype(float)
    Yc = y.astype(float)
    Zc = z.astype(float)
    X3, Y3, Z3 = np.meshgrid(Xc, Yc, Zc, indexing='ij')
    grid_coords = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    c_mean = grid_coords.mean(axis=0, keepdims=True)
    c_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - c_mean) / (c_std + 1e-12)
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32, device=device, requires_grad=True)

    dz_half = dz / 2.0
    cell_grid = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)])
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    print("Assembling sensitivity G ...")
    t0 = time.time()
    G = construct_sensitivity_matrix_G_torch(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)
    print(f"G shape = {tuple(G.shape)}, time = {time.time() - t0:.2f}s")

    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true_vec = rho_true_vec.to(device)

    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)

    sigma = NOISE_LEVEL * gz_true.std()
    noise = sigma * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    Wd = 1.0 / sigma
    model_rng_state = capture_rng_state()

    cfg = dict(gamma=GAMMA, epochs=EPOCHS, lr=LR)
    cfg.update(
        use_early_stopping=USE_EARLY_STOPPING,
        early_stop_min_epochs=EARLY_STOP_MIN_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_target=EARLY_STOP_TARGET,
        early_stop_tol=EARLY_STOP_TOL,
        early_stop_overfit_patience=EARLY_STOP_OVERFIT_PATIENCE,
    )

    results = {}
    for encoding_type in ENCODING_ORDER:
        restore_rng_state(model_rng_state)
        print(f"\n▶  Encoding strategy: {encoding_type}")
        model = DensityContrastINR(
            encoding_type=encoding_type, hidden=HIDDEN, depth=DEPTH,
            rho_abs_max=RHO_ABS_MAX,
            **ENCODING_CONFIGS[encoding_type]
        ).to(device)
        opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        hist = train_inr(model, opt, coords_norm, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, cfg)

        with torch.no_grad():
            m_inv = model(coords_norm).view(-1)
            gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)

        results[encoding_type] = {
            'm_inv': m_inv.detach().cpu().numpy().reshape(Nx, Ny, Nz),
            'gz_pred': gz_pred.detach().cpu().numpy(),
            'hist': hist,
            'rms_rho': torch.sqrt(torch.mean((m_inv - rho_true_vec.to(device)) ** 2)).item(),
            'rms_gz': torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item() * 1e5,
        }

    def get_axes_coords():
        x1d = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
        y1d = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
        z1d = grid_coords[:, 2].reshape(Nx, Ny, Nz)[0, 0, :]
        return x1d, y1d, z1d

    x1d, y1d, z1d = get_axes_coords()
    block_boundaries = get_block_boundaries(Nx, Ny, Nz)

    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    tru = rho_true_3d.cpu().numpy()
    inv_pos = results['positional']['m_inv']
    inv_hash = results['hash']['m_inv']

    tru_max = tru.max()
    inv_max = INV_VMAX

    fig1, axes = plt.subplots(4, 3, figsize=(16, 20))
    x1d, y1d, z1d = get_axes_coords()

    # cell-edge limits
    x_edge_min, x_edge_max = x1d[0] - dx/2, x1d[-1] + dx/2
    y_edge_min, y_edge_max = y1d[0] - dy/2, y1d[-1] + dy/2
    z_edge_min, z_edge_max = z1d[0] - dz/2, z1d[-1] + dz/2

    # use edges for all extents
    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    # for depth plots keep depth increasing downward by reversing z limits
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]


    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 0].set_title(f"True Model XY @ z≈{z1d[iz]:.0f} m", fontsize=TITLE_FONTSIZE)
    cbar = fig1.colorbar(im, ax=axes[0, 0], label='kg/m³', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    im = axes[0, 1].imshow(tru[:, iy, :].T, origin='upper', extent=extent_xz,
                           aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 1].set_title(f"True Model XZ @ y≈{y1d[iy]:.0f} m", fontsize=TITLE_FONTSIZE)

    im = axes[0, 2].imshow(tru[ix, :, :].T, origin='upper', extent=extent_yz,
                           aspect='auto', vmin=0, vmax=tru_max, cmap=CMAP)
    axes[0, 2].set_title(f"True Model YZ @ x≈{x1d[ix]:.0f} m", fontsize=TITLE_FONTSIZE)

    model_rows = [
        ('positional', inv_pos, 1),
        ('hash', inv_hash, 2),
    ]
    for encoding_type, inv, row in model_rows:
        title = encoding_label(encoding_type)

        im = axes[row, 0].imshow(inv[:, :, iz].T, origin='lower', extent=extent_xy,
                                 aspect='auto', vmin=0, vmax=inv_max, cmap=CMAP)
        axes[row, 0].set_title(f"Recovered Model XY, {title}", fontsize=TITLE_FONTSIZE)
        add_xy_outline(axes[row, 0], block_boundaries, x, y, dx, dy, iz)
        cbar = fig1.colorbar(im, ax=axes[row, 0], label='kg/m³', fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

        im = axes[row, 1].imshow(inv[:, iy, :].T, origin='upper', extent=extent_xz,
                                 aspect='auto', vmin=0, vmax=inv_max, cmap=CMAP)
        axes[row, 1].set_title(f"Recovered Model XZ, {title}", fontsize=TITLE_FONTSIZE)
        add_xz_outline(axes[row, 1], block_boundaries, x, z, dx, dz, iy)

        im = axes[row, 2].imshow(inv[ix, :, :].T, origin='upper', extent=extent_yz,
                                 aspect='auto', vmin=0, vmax=inv_max, cmap=CMAP)
        axes[row, 2].set_title(f"Recovered Model YZ, {title}", fontsize=TITLE_FONTSIZE)
        add_yz_outline(axes[row, 2], block_boundaries, y, z, dy, dz, ix)

    for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2], axes[2, 1], axes[2, 2]]:
        ax.set_aspect(1.0)
    for ax in [axes[0, 0], axes[1, 0], axes[2, 0]]:
        ax.set_aspect(1.0)

    for row in range(3):
        style_axes(axes[row, 0], 'x (m)', 'y (m)')
        style_axes(axes[row, 1], 'x (m)', 'Depth (m)')
        style_axes(axes[row, 2], 'y (m)', 'Depth (m)')

    def to_mgal(g):
        return 1e5 * g.detach().cpu().numpy()
    obs_mgal = to_mgal(gz_obs)
    res_pos_mgal = obs_mgal - 1e5 * results['positional']['gz_pred']
    res_hash_mgal = obs_mgal - 1e5 * results['hash']['gz_pred']

    obs_x = obs[:, 0].cpu().numpy()
    obs_y = obs[:, 1].cpu().numpy()

    axes[3, 0].plot(results['positional']['hist']['gravity'], label='Positional', color='tab:red')
    axes[3, 0].plot(results['hash']['hist']['gravity'], label='Hash', color='black')
    axes[3, 0].set_title('Training Convergence by Encoding', fontsize=TITLE_FONTSIZE)
    axes[3, 0].set_yscale('log')
    axes[3, 0].grid(True, which='both', ls='--', alpha=0.3)
    axes[3, 0].legend(fontsize=LEGEND_FONTSIZE)
    style_axes(axes[3, 0], 'Epoch', 'Loss = mean((residual / sigma)^2)')

    vmax_res = max(np.abs(res_pos_mgal).max(), np.abs(res_hash_mgal).max())
    sc = axes[3, 1].scatter(obs_x, obs_y, c=res_pos_mgal, s=80, cmap=CMAP,
                            vmin=-vmax_res, vmax=vmax_res, marker='o', edgecolors='none')
    axes[3, 1].set_title(
        f"Data Misfit, Positional Encoding (RMS = {np.sqrt(np.mean(res_pos_mgal**2)):.3f} mGal)",
        fontsize=TITLE_FONTSIZE,
    )
    cbar = fig1.colorbar(sc, ax=axes[3, 1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    sc = axes[3, 2].scatter(obs_x, obs_y, c=res_hash_mgal, s=80, cmap=CMAP,
                            vmin=-vmax_res, vmax=vmax_res, marker='o', edgecolors='none')
    axes[3, 2].set_title(
        f"Data Misfit, Hash Encoding (RMS = {np.sqrt(np.mean(res_hash_mgal**2)):.3f} mGal)",
        fontsize=TITLE_FONTSIZE,
    )
    cbar = fig1.colorbar(sc, ax=axes[3, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    for ax in [axes[3, 1], axes[3, 2]]:
        style_axes(ax, 'x (m)', 'y (m)')
        ax.set_aspect('equal')

    fig1.tight_layout()
    fig1.savefig('plots/BlockModel_EncodingComparison.png', dpi=FIG_DPI)
    plt.close(fig1)

    for encoding_type in ENCODING_ORDER:
        label = encoding_label(encoding_type)
        hist = results[encoding_type]['hist']
        print(
            f"{label} RMS density-contrast error ≈ {results[encoding_type]['rms_rho']:.2f} kg/m^3"
        )
        print(
            f"{label} RMS data misfit ≈ {results[encoding_type]['rms_gz']:.3f} mGal | "
            f"best epoch = {hist['best_epoch']} | "
            f"best loss = {hist['best_weighted_mse']:.3f}"
        )

if __name__ == '__main__':
    run()
