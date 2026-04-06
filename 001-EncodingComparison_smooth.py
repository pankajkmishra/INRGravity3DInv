"""
01-GRF-MLP_vs_PosMLP.py
=======================
Compare no encoding, positional encoding, and hash encoding for 3-D gravity
inversion of a Gaussian random field density model.

This script follows the newer experiment layout used in 001/002:
top-level configuration blocks, small helper functions, explicit plotting
constants, and a single run() entry point.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# --- Unit conversion ---------------------------------------------------
MGAL_PER_MPS2 = 1e5
RHO_GC_TO_KGM3 = 1000.0

# --- Random seed -------------------------------------------------------
SEED = 41

# --- Grid / domain -----------------------------------------------------
NX = 40
NY = 40
NZ = 20
DX = 500.0
DY = 500.0
DZ = 500.0

# --- Density model -----------------------------------------------------
RHO_MIN_GCC = 1.6
RHO_MAX_GCC = 3.5

# --- Gaussian random field --------------------------------------------
GRF_LAMBDA = 5000.0
GRF_NU = 1.5
GRF_SIGMA = 2.0

# --- Noise level -------------------------------------------------------
NOISE_LEVEL = 0.01

# --- Training / optimisation -------------------------------------------
EPOCHS = 500
LR = 1e-3

# --- Early stopping ----------------------------------------------------
# Stop near the expected noise floor to reduce overfitting to 1% noise.
USE_EARLY_STOPPING = True
EARLY_STOP_MIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
EARLY_STOP_TARGET = 1.0
EARLY_STOP_TOL = 0.05
EARLY_STOP_OVERFIT_PATIENCE = 5

# --- INR network -------------------------------------------------------
HIDDEN = 256
DEPTH = 4
RHO_ABS_MAX = 3.5

# --- Encoding comparison -----------------------------------------------
NUM_FREQS = 2
HASH_CONFIG = dict(
    n_levels=2,
    n_features_per_level=2,
    log2_hashmap_size=17,
    base_resolution=4,
    finest_resolution=128,
)

# --- Plotting -----------------------------------------------------------
CMAP = 'Spectral_r'
FIG_DPI = 300
TITLE_FONTSIZE = 15
LABEL_FONTSIZE = 14
TICK_FONTSIZE = 12
COLORBAR_FONTSIZE = 13


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for 3-D coordinates."""

    def __init__(self, num_freqs=4, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        self.out_dim = input_dim * (1 + 2 * num_freqs) if include_input else input_dim * 2 * num_freqs

    def forward(self, x):
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(parts, dim=-1)


class IdentityEncoding(nn.Module):
    """Pass-through baseline with no spatial encoding."""

    def __init__(self, input_dim=3):
        super().__init__()
        self.out_dim = input_dim

    def forward(self, x):
        return x


class HashEncoding(nn.Module):
    """Multi-resolution hash encoding adapted from 001-EncodingComparisons.py."""

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

        self.hash_tables = nn.ModuleList([
            nn.Embedding(self.hashmap_size, n_features_per_level)
            for _ in range(n_levels)
        ])
        for tbl in self.hash_tables:
            nn.init.uniform_(tbl.weight, -1e-4, 1e-4)

        self.register_buffer(
            'primes', torch.tensor([1, 2654435761, 805459861], dtype=torch.long))

    def _hash(self, coords_int):
        result = torch.zeros(coords_int.shape[:-1], dtype=torch.long, device=coords_int.device)
        for dim in range(self.input_dim):
            result ^= coords_int[..., dim] * self.primes[dim]
        return result % self.hashmap_size

    def forward(self, x):
        x_min = x.min(dim=0, keepdim=True).values
        x_max = x.max(dim=0, keepdim=True).values
        x_scaled = (x - x_min) / (x_max - x_min + 1e-8)

        outputs = []
        for level in range(self.n_levels):
            resolution = int(self.base_resolution * (self.growth_factor ** level))
            x_grid = x_scaled * resolution
            x_floor = torch.floor(x_grid).long()
            x_frac = x_grid - x_floor.float()

            corners = []
            for dz in (0, 1):
                for dy in (0, 1):
                    for dx in (0, 1):
                        corners.append(x_floor + torch.tensor([dx, dy, dz], device=x.device))
            corners = torch.stack(corners, dim=1)

            indices = self._hash(corners)
            features = self.hash_tables[level](indices)

            wx, wy, wz = x_frac[:, 0:1], x_frac[:, 1:2], x_frac[:, 2:3]
            weights = torch.stack([
                (1 - wx) * (1 - wy) * (1 - wz), wx * (1 - wy) * (1 - wz),
                (1 - wx) * wy * (1 - wz),       wx * wy * (1 - wz),
                (1 - wx) * (1 - wy) * wz,       wx * (1 - wy) * wz,
                (1 - wx) * wy * wz,             wx * wy * wz,
            ], dim=1)
            outputs.append((weights * features).sum(dim=1))

        return torch.cat(outputs, dim=-1)


def create_encoding(encoding_type, **kwargs):
    if encoding_type == 'none':
        return IdentityEncoding(input_dim=kwargs.get('input_dim', 3))
    if encoding_type == 'positional':
        return PositionalEncoding(
            num_freqs=kwargs.get('num_freqs', 2),
            include_input=kwargs.get('include_input', True),
        )
    if encoding_type == 'hash':
        return HashEncoding(
            n_levels=kwargs.get('n_levels', 16),
            n_features_per_level=kwargs.get('n_features_per_level', 2),
            log2_hashmap_size=kwargs.get('log2_hashmap_size', 19),
            base_resolution=kwargs.get('base_resolution', 16),
            finest_resolution=kwargs.get('finest_resolution', 512),
        )
    raise ValueError(f"Unknown encoding type: {encoding_type}")


class DensityINR(nn.Module):
    """Implicit density model with pluggable spatial encoding."""

    def __init__(self, encoding_type='positional', hidden=256, depth=4,
                 rho_min_gcc=1.6, rho_max_gcc=3.5, **encoding_kwargs):
        super().__init__()
        self.rho_min_gcc = float(rho_min_gcc)
        self.rho_max_gcc = float(rho_max_gcc)
        self.encoding = create_encoding(encoding_type, **encoding_kwargs)
        in_dim = self.encoding.out_dim

        layers = [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.01)]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden, hidden), nn.LeakyReLU(0.01)]
        layers += [nn.Linear(hidden, 1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoding(x)
        rho_norm = self.net(encoded)
        return self.rho_min_gcc + rho_norm * (self.rho_max_gcc - self.rho_min_gcc)


def set_seed(seed: int):
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


def a_integral_torch(x, y, z):
    eps = 1e-20
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + r) + eps) +
             y * torch.log(torch.abs(x + r) + eps) -
             z * torch.atan2(x * y, z * r + eps))


@torch.inference_mode()
def construct_sensitivity_matrix_G(cell_grid, data_points, d1, d2, device):
    gamma = 6.67430e-11
    cx = cell_grid[:, 0].unsqueeze(0)
    cy = cell_grid[:, 1].unsqueeze(0)
    cz = cell_grid[:, 2].unsqueeze(0)
    czh = cell_grid[:, 3].unsqueeze(0)
    ox = data_points[:, 0].unsqueeze(1)
    oy = data_points[:, 1].unsqueeze(1)
    oz = data_points[:, 2].unsqueeze(1)

    x2, x1 = (cx + d1 / 2) - ox, (cx - d1 / 2) - ox
    y2, y1 = (cy + d2 / 2) - oy, (cy - d2 / 2) - oy
    z2, z1 = (cz + czh) - oz, (cz - czh) - oz

    a = (a_integral_torch(x2, y2, z2) - a_integral_torch(x2, y2, z1) -
         a_integral_torch(x2, y1, z2) + a_integral_torch(x2, y1, z1) -
         a_integral_torch(x1, y2, z2) + a_integral_torch(x1, y2, z1) +
         a_integral_torch(x1, y1, z2) - a_integral_torch(x1, y1, z1))
    return (gamma * a).to(device)


def generate_grf_torch(nx, ny, nz, dx, dy, dz, lam, nu, sigma, device):
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * torch.pi
    kx_grid, ky_grid, kz_grid = torch.meshgrid(kx, ky, kz, indexing='ij')
    k2 = kx_grid**2 + ky_grid**2 + kz_grid**2
    power = (k2 + (1 / lam**2))**(-nu - 1.5)
    power[0, 0, 0] = 0

    noise = torch.randn(nx, ny, nz, dtype=torch.complex64, device=device)
    fourier_field = noise * torch.sqrt(power)
    field = torch.real(torch.fft.ifftn(fourier_field))
    field = (field - field.mean()) / (field.std() + 1e-9)
    return sigma * field


def train_inr(model, optimizer, coords_norm, G, gz_obs, sigma_noise, cfg):
    history = {"loss": []}
    wd = 1.0 / sigma_noise
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
        optimizer.zero_grad()
        rho_pred_gcc = model(coords_norm).view(-1)
        rho_pred_kgm3 = rho_pred_gcc * RHO_GC_TO_KGM3
        gz_pred = torch.matmul(G, rho_pred_kgm3.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        loss = torch.mean((wd * residual) ** 2)
        loss.backward()
        optimizer.step()

        history['loss'].append(float(loss.item()))

        if use_early_stopping:
            weighted_mse = float(loss.item())
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
                        f"weighted MSE = {weighted_mse:.3f} | "
                        f"best epoch = {best_epoch}"
                    )
                    break

                if overfit_count >= overfit_patience:
                    print(
                        f"Early stopping at epoch {ep:4d} | "
                        f"reason = below noise floor | "
                        f"weighted MSE = {weighted_mse:.3f} | "
                        f"best epoch = {best_epoch}"
                    )
                    break

        if ep % 50 == 0 or ep == cfg['epochs'] - 1:
            weighted_mse = history['loss'][-1]
            print(f"Epoch {ep:4d} | loss {weighted_mse:.3e}")

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        history['best_epoch'] = best_epoch
        history['best_weighted_mse'] = best_weighted_mse
    else:
        history['best_epoch'] = len(history['loss']) - 1
        history['best_weighted_mse'] = history['loss'][-1]

    return history


@torch.no_grad()
def evaluate_model(model, coords_norm, G, gz_obs):
    rho_pred_gcc = model(coords_norm).view(-1)
    rho_pred_kgm3 = rho_pred_gcc * RHO_GC_TO_KGM3
    gz_pred = torch.matmul(G, rho_pred_kgm3.unsqueeze(1)).squeeze(1)
    rms_gz = float(torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item()) * MGAL_PER_MPS2
    return rho_pred_gcc, gz_pred, rms_gz


def plot_comparison_results(rho_true_gcc, rho_none_gcc, rho_pos_gcc, rho_hash_gcc,
                            gz_obs, gz_pred_none, gz_pred_pos, gz_pred_hash,
                            hist_none, hist_pos, hist_hash,
                            grid_coords, obs_points,
                            dx, dy, nz, save_path):
    iz = nz // 2

    tru = rho_true_gcc.view(NX, NY, NZ).cpu().numpy()
    none = rho_none_gcc.view(NX, NY, NZ).detach().cpu().numpy()
    pos = rho_pos_gcc.view(NX, NY, NZ).detach().cpu().numpy()
    hsh = rho_hash_gcc.view(NX, NY, NZ).detach().cpu().numpy()

    x1d = grid_coords[:, 0].reshape(NX, NY, NZ)[:, 0, 0]
    y1d = grid_coords[:, 1].reshape(NX, NY, NZ)[0, :, 0]
    x_edge_min, x_edge_max = x1d[0] - dx / 2, x1d[-1] + dx / 2
    y_edge_min, y_edge_max = y1d[0] - dy / 2, y1d[-1] + dy / 2
    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]

    fig, axes = plt.subplots(2, 4, figsize=(24, 11))

    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=RHO_MIN_GCC, vmax=RHO_MAX_GCC,
                           cmap=CMAP)
    axes[0, 0].set_title(f"True Model, XY Slice (z-index = {iz})", fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(im, ax=axes[0, 0], label='g/cm^3', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label('g/cm^3', fontsize=COLORBAR_FONTSIZE)

    im = axes[0, 1].imshow(none[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=RHO_MIN_GCC, vmax=RHO_MAX_GCC,
                           cmap=CMAP)
    axes[0, 1].set_title('Recovered Model, No Encoding', fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(im, ax=axes[0, 1], label='g/cm^3', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label('g/cm^3', fontsize=COLORBAR_FONTSIZE)

    im = axes[0, 2].imshow(pos[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=RHO_MIN_GCC, vmax=RHO_MAX_GCC,
                           cmap=CMAP)
    axes[0, 2].set_title('Recovered Model, Positional Encoding', fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(im, ax=axes[0, 2], label='g/cm^3', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label('g/cm^3', fontsize=COLORBAR_FONTSIZE)

    im = axes[0, 3].imshow(hsh[:, :, iz].T, origin='lower', extent=extent_xy,
                           aspect='equal', vmin=RHO_MIN_GCC, vmax=RHO_MAX_GCC,
                           cmap=CMAP)
    axes[0, 3].set_title('Recovered Model, Hash Encoding', fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(im, ax=axes[0, 3], label='g/cm^3', fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)
    cbar.set_label('g/cm^3', fontsize=COLORBAR_FONTSIZE)

    obs_mgal = MGAL_PER_MPS2 * gz_obs.detach().cpu().numpy()
    none_mgal = MGAL_PER_MPS2 * gz_pred_none.detach().cpu().numpy()
    pos_mgal = MGAL_PER_MPS2 * gz_pred_pos.detach().cpu().numpy()
    hash_mgal = MGAL_PER_MPS2 * gz_pred_hash.detach().cpu().numpy()
    res_none_mgal = obs_mgal - none_mgal
    res_pos_mgal = obs_mgal - pos_mgal
    res_hash_mgal = obs_mgal - hash_mgal

    obs_x = obs_points[:, 0]
    obs_y = obs_points[:, 1]
    v_res = max(abs(res_none_mgal).max(), abs(res_pos_mgal).max(), abs(res_hash_mgal).max())

    axes[1, 0].plot(hist_none['loss'], label='No encoding', color='tab:blue')
    axes[1, 0].plot(hist_pos['loss'], label='Positional', color='tab:red')
    axes[1, 0].plot(hist_hash['loss'], label='Hash', color='black')
    axes[1, 0].set_title('Training Convergence by Encoding', fontsize=TITLE_FONTSIZE)
    axes[1, 0].set_xlabel('Epoch', fontsize=LABEL_FONTSIZE)
    axes[1, 0].set_ylabel('Loss = mean((residual / sigma_noise)^2)', fontsize=LABEL_FONTSIZE)
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, which='both', ls='--', alpha=0.3)
    axes[1, 0].legend(fontsize=TICK_FONTSIZE)
    axes[1, 0].tick_params(labelsize=TICK_FONTSIZE)

    sc = axes[1, 1].scatter(obs_x, obs_y, c=res_none_mgal, s=18, cmap=CMAP,
                            vmin=-v_res, vmax=v_res, marker='o', edgecolors='none')
    axes[1, 1].set_title(f'Data Misfit, No Encoding (RMS = {np.sqrt(np.mean(res_none_mgal**2)):.3f} mGal)', fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(sc, ax=axes[1, 1], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    sc = axes[1, 2].scatter(obs_x, obs_y, c=res_pos_mgal, s=18, cmap=CMAP,
                            vmin=-v_res, vmax=v_res, marker='o', edgecolors='none')
    axes[1, 2].set_title(f'Data Misfit, Positional Encoding (RMS = {np.sqrt(np.mean(res_pos_mgal**2)):.3f} mGal)', fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(sc, ax=axes[1, 2], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    sc = axes[1, 3].scatter(obs_x, obs_y, c=res_hash_mgal, s=18, cmap=CMAP,
                            vmin=-v_res, vmax=v_res, marker='o', edgecolors='none')
    axes[1, 3].set_title(f'Data Misfit, Hash Encoding (RMS = {np.sqrt(np.mean(res_hash_mgal**2)):.3f} mGal)', fontsize=TITLE_FONTSIZE)
    cbar = fig.colorbar(sc, ax=axes[1, 3], fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FONTSIZE)

    for ax in axes[0, :]:
        ax.set_xlabel('x (m)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('y (m)', fontsize=LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
    for ax in axes[1, 1:]:
        ax.set_xlabel('x (m)', fontsize=LABEL_FONTSIZE)
        ax.set_ylabel('y (m)', fontsize=LABEL_FONTSIZE)
        ax.tick_params(labelsize=TICK_FONTSIZE)
    for ax in axes[0, :]:
        ax.set_aspect('equal')
    for ax in axes[1, 1:]:
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f"Saved {save_path}")


def run():
    set_seed(SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    x = np.linspace(0.0, (NX - 1) * DX, NX)
    y = np.linspace(0.0, (NY - 1) * DY, NY)
    z = np.linspace(0.0, (NZ - 1) * DZ, NZ)
    x3, y3, z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=1)

    coords_mean = grid_coords.mean(axis=0, keepdims=True)
    coords_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - coords_mean) / (coords_std + 1e-12)
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32, device=device)

    dz_half = DZ / 2.0
    cell_grid = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    xx_obs, yy_obs = np.meshgrid(x, y, indexing='ij')
    obs_points = np.column_stack([xx_obs.ravel(), yy_obs.ravel(), np.zeros(xx_obs.size)])
    obs = torch.tensor(obs_points, dtype=torch.float32, device=device)

    print('Assembling sensitivity G ...')
    t0 = time.time()
    G = construct_sensitivity_matrix_G(cell_grid, obs, DX, DY, device)
    G = G.clone().detach().requires_grad_(False)
    print(f"G shape = {tuple(G.shape)}, time = {time.time() - t0:.2f}s")

    rho_true_3d = generate_grf_torch(NX, NY, NZ, DX, DY, DZ,
                                     GRF_LAMBDA, GRF_NU, GRF_SIGMA, device)
    rho_min = rho_true_3d.min()
    rho_max = rho_true_3d.max()
    rho_true_3d = RHO_MIN_GCC + (rho_true_3d - rho_min) * ((RHO_MAX_GCC - RHO_MIN_GCC) / (rho_max - rho_min + 1e-12))
    rho_true = rho_true_3d.view(-1)

    with torch.no_grad():
        gz_true = torch.matmul(G, (rho_true * RHO_GC_TO_KGM3).unsqueeze(1)).squeeze(1)

    sigma_noise = NOISE_LEVEL * gz_true.std()
    noise = sigma_noise * torch.randn_like(gz_true)
    gz_obs = gz_true + noise

    cfg = dict(epochs=EPOCHS, lr=LR)
    cfg.update(
        use_early_stopping=USE_EARLY_STOPPING,
        early_stop_min_epochs=EARLY_STOP_MIN_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_target=EARLY_STOP_TARGET,
        early_stop_tol=EARLY_STOP_TOL,
        early_stop_overfit_patience=EARLY_STOP_OVERFIT_PATIENCE,
    )

    print('\nNo-encoding INR')
    model_none = DensityINR(
        encoding_type='none',
        hidden=HIDDEN,
        depth=DEPTH,
        rho_min_gcc=RHO_MIN_GCC,
        rho_max_gcc=RHO_MAX_GCC,
    ).to(device)
    opt_none = torch.optim.Adam(model_none.parameters(), lr=cfg['lr'])
    hist_none = train_inr(model_none, opt_none, coords_norm, G, gz_obs, sigma_noise, cfg)
    rho_none, gz_pred_none, rms_none = evaluate_model(model_none, coords_norm, G, gz_obs)

    print('\nPositional-encoding INR')
    model_pos = DensityINR(
        encoding_type='positional',
        num_freqs=NUM_FREQS,
        hidden=HIDDEN,
        depth=DEPTH,
        rho_min_gcc=RHO_MIN_GCC,
        rho_max_gcc=RHO_MAX_GCC,
    ).to(device)
    opt_pos = torch.optim.Adam(model_pos.parameters(), lr=cfg['lr'])
    hist_pos = train_inr(model_pos, opt_pos, coords_norm, G, gz_obs, sigma_noise, cfg)
    rho_pos, gz_pred_pos, rms_pos = evaluate_model(model_pos, coords_norm, G, gz_obs)

    print('\nHash-encoding INR')
    model_hash = DensityINR(
        encoding_type='hash',
        hidden=HIDDEN,
        depth=DEPTH,
        rho_min_gcc=RHO_MIN_GCC,
        rho_max_gcc=RHO_MAX_GCC,
        **HASH_CONFIG,
    ).to(device)
    opt_hash = torch.optim.Adam(model_hash.parameters(), lr=cfg['lr'])
    hist_hash = train_inr(model_hash, opt_hash, coords_norm, G, gz_obs, sigma_noise, cfg)
    rho_hash, gz_pred_hash, rms_hash = evaluate_model(model_hash, coords_norm, G, gz_obs)

    plot_comparison_results(
        rho_true_gcc=rho_true.detach().cpu(),
        rho_none_gcc=rho_none,
        rho_pos_gcc=rho_pos,
        rho_hash_gcc=rho_hash,
        gz_obs=gz_obs,
        gz_pred_none=gz_pred_none,
        gz_pred_pos=gz_pred_pos,
        gz_pred_hash=gz_pred_hash,
        hist_none=hist_none,
        hist_pos=hist_pos,
        hist_hash=hist_hash,
        grid_coords=grid_coords,
        obs_points=obs_points,
        dx=DX,
        dy=DY,
        nz=NZ,
        save_path='plots/EncodingComparisonSmooth.png',
    )

    print(
        f"No-encoding INR RMS misfit : {rms_none:.3f} mGal | "
        f"best epoch = {hist_none['best_epoch']} | "
        f"best loss = {hist_none['best_weighted_mse']:.3f}"
    )
    print(
        f"Positional INR RMS misfit  : {rms_pos:.3f} mGal | "
        f"best epoch = {hist_pos['best_epoch']} | "
        f"best loss = {hist_pos['best_weighted_mse']:.3f}"
    )
    print(
        f"Hash INR RMS misfit        : {rms_hash:.3f} mGal | "
        f"best epoch = {hist_hash['best_epoch']} | "
        f"best loss = {hist_hash['best_weighted_mse']:.3f}"
    )


if __name__ == '__main__':
    run()
