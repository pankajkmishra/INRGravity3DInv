"""
003-ImplictvsExplicitRegularisation.py
=====================================
Compare implicit regularization from an INR against explicit smoothness
and TV regularization on the block model.

The INR uses the same medium-grid architecture as the grid-size test:
cell size 50 m, positional encoding with 2 frequency bands, and hidden
layers [48, 48, 24], so the INR and voxel parameter counts are similar.
"""

import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


MGAL_PER_MPS2 = 1e5


# --- Random seed -------------------------------------------------------
SEED = 42
DATA_SEED = SEED

# --- Grid / domain -----------------------------------------------------
DX = 50.0
DY = 50.0
DZ = 50.0
X_MAX = 1000.0
Y_MAX = 1000.0
Z_MAX = 500.0

# --- Block model -------------------------------------------------------
RHO_BG = 0.0
RHO_BLK = 400.0
RHO_ABS_MAX = 600.0

# --- Noise level -------------------------------------------------------
NOISE_LEVEL = 0.01

# --- INR training ------------------------------------------------------
INR_EPOCHS = 500
INR_LR = 1e-2
INR_NUM_FREQS = 2
INR_HIDDEN_SIZES = [48, 48, 24]
GAMMA = 1.0
INR_PROGRESS_EVERY = 25

# --- Early stopping ----------------------------------------------------
USE_EARLY_STOPPING = True
EARLY_STOP_MIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
EARLY_STOP_TARGET = 1.0
EARLY_STOP_TOL = 0.05
EARLY_STOP_OVERFIT_PATIENCE = 5

# --- Classical inversion -----------------------------------------------
CG_MAX_ITER = 800
CG_TOL = 5e-5
SMOOTH_CFG = dict(z0=50.0, beta=1.5, alpha_s=1e-2, alpha_x=1.0, alpha_y=1.0, alpha_z=1.0)
TV_CFG = dict(z0=50.0, beta=1.5, alpha_s=1e-2, alpha_tv=0.5, irls_iters=15, irls_eps=1e-3)
REG_SCALE_MIN = 2.0 ** -8
REG_SCALE_MAX = 2.0 ** 8
REG_SEARCH_STEPS = 8

# --- Plotting -----------------------------------------------------------
CMAP = 'Spectral_r'
FIG_DPI = 300
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


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


def a_integral_torch(x, y, z):
    eps = 1e-20
    radius = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + radius) + eps) +
             y * torch.log(torch.abs(x + radius) + eps) -
             z * torch.atan2(x * y, z * radius + eps))


@torch.inference_mode()
def construct_sensitivity_matrix_G(cell_grid, data_points, dx, dy, device):
    gamma = 6.67430e-11
    cx = cell_grid[:, 0].unsqueeze(0)
    cy = cell_grid[:, 1].unsqueeze(0)
    cz = cell_grid[:, 2].unsqueeze(0)
    czh = cell_grid[:, 3].unsqueeze(0)
    ox = data_points[:, 0].unsqueeze(1)
    oy = data_points[:, 1].unsqueeze(1)
    oz = data_points[:, 2].unsqueeze(1)

    x2, x1 = (cx + dx / 2) - ox, (cx - dx / 2) - ox
    y2, y1 = (cy + dy / 2) - oy, (cy - dy / 2) - oy
    z2, z1 = (cz + czh) - oz, (cz - czh) - oz

    a = (a_integral_torch(x2, y2, z2) - a_integral_torch(x2, y2, z1) -
         a_integral_torch(x2, y1, z2) + a_integral_torch(x2, y1, z1) -
         a_integral_torch(x1, y2, z2) + a_integral_torch(x1, y2, z1) +
         a_integral_torch(x1, y1, z2) - a_integral_torch(x1, y1, z1))
    return (gamma * a).to(device)


def make_block_model(nx, ny, nz, rho_bg=0.0, rho_blk=400.0):
    model = torch.full((nx, ny, nz), rho_bg)
    for index in range(7):
        z_idx = 1 + index
        y_start, y_end = 11 - index, 16 - index
        x_start, x_end = 7, 13
        if 0 <= z_idx < nz:
            ys, ye = max(0, y_start), min(ny, y_end)
            xs, xe = max(0, x_start), min(nx, x_end)
            model[xs:xe, ys:ye, z_idx] = rho_blk
    return model.view(-1), model


class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=2, include_input=True, input_dim=3):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))
        if include_input:
            self.out_dim = input_dim * (1 + 2 * num_freqs)
        else:
            self.out_dim = input_dim * 2 * num_freqs

    def forward(self, x):
        parts = [x] if self.include_input else []
        for freq in self.freqs:
            parts += [torch.sin(freq * x), torch.cos(freq * x)]
        return torch.cat(parts, dim=-1)


class DensityContrastINR(nn.Module):
    def __init__(self, hidden_sizes, num_freqs=2, rho_abs_max=600.0):
        super().__init__()
        self.encoding = PositionalEncoding(num_freqs=num_freqs)
        layers = []
        in_dim = self.encoding.out_dim
        for hidden in hidden_sizes:
            layers += [nn.Linear(in_dim, hidden), nn.LeakyReLU(0.01)]
            in_dim = hidden
        layers += [nn.Linear(in_dim, 1)]
        self.network = nn.Sequential(*layers)
        self.rho_abs_max = float(rho_abs_max)

    def forward(self, x):
        encoded = self.encoding(x)
        output = self.network(encoded)
        return self.rho_abs_max * torch.tanh(output)


def count_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg):
    history = {'loss': []}
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
    best_safe_gap = float('inf')
    best_safe_epoch = -1
    best_safe_weighted_mse = None
    best_safe_state = None
    in_band_count = 0
    overfit_count = 0
    lower_bound = target - tol
    progress_every = cfg.get('progress_every', 0)

    for epoch in range(cfg['epochs']):
        optimizer.zero_grad()
        rho_pred = model(coords_norm).view(-1)
        gz_pred = torch.matmul(G, rho_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        loss = cfg['gamma'] * torch.mean((wd * residual) ** 2)
        loss.backward()
        optimizer.step()
        history['loss'].append(float(loss.item()))
        weighted_mse = float(loss.item() / cfg['gamma'])

        if progress_every and ((epoch + 1) == 1 or (epoch + 1) % progress_every == 0 or (epoch + 1) == cfg['epochs']):
            print(f'  INR epoch {epoch + 1:4d}/{cfg["epochs"]}: weighted MSE = {weighted_mse:.4f}')

        if use_early_stopping:
            gap = abs(weighted_mse - target)
            if gap < best_gap:
                best_gap = gap
                best_epoch = epoch
                best_weighted_mse = weighted_mse
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if weighted_mse >= lower_bound and gap < best_safe_gap:
                best_safe_gap = gap
                best_safe_epoch = epoch
                best_safe_weighted_mse = weighted_mse
                best_safe_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if epoch + 1 >= min_epochs:
                if target - tol <= weighted_mse <= target + tol:
                    in_band_count += 1
                else:
                    in_band_count = 0

                if weighted_mse < target - tol:
                    overfit_count += 1
                else:
                    overfit_count = 0

                if in_band_count >= patience or overfit_count >= overfit_patience:
                    print(f'  INR early stop at epoch {epoch + 1}: weighted MSE = {weighted_mse:.4f}')
                    break

    if use_early_stopping and best_safe_state is not None:
        model.load_state_dict(best_safe_state)
        history['best_epoch'] = best_safe_epoch
        history['best_weighted_mse'] = best_safe_weighted_mse
    elif use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        history['best_epoch'] = best_epoch
        history['best_weighted_mse'] = best_weighted_mse
    else:
        history['best_epoch'] = len(history['loss']) - 1
        history['best_weighted_mse'] = history['loss'][-1] / cfg['gamma']

    return history


def idx_flat(i, j, k, ny, nz):
    return i * ny * nz + j * nz + k


def build_grad_ops_sparse(nx, ny, nz, dx, dy, dz, device):
    operators = []
    directions = [
        (dx, lambda i, j, k: (i + 1, j, k)),
        (dy, lambda i, j, k: (i, j + 1, k)),
        (dz, lambda i, j, k: (i, j, k + 1)),
    ]
    for spacing, shift in directions:
        rows, cols, vals = [], [], []
        row = 0
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    si, sj, sk = shift(i, j, k)
                    if si < nx and sj < ny and sk < nz:
                        c0 = idx_flat(i, j, k, ny, nz)
                        c1 = idx_flat(si, sj, sk, ny, nz)
                        rows += [row, row]
                        cols += [c1, c0]
                        vals += [1.0 / spacing, -1.0 / spacing]
                        row += 1
        operator = torch.sparse_coo_tensor(
            indices=torch.tensor([rows, cols], dtype=torch.long),
            values=torch.tensor(vals, dtype=torch.float32),
            size=(row, nx * ny * nz),
            device=device,
        ).coalesce()
        operators.append(operator)
    return operators


def depth_weights(grid_coords_t, z0, beta, normalize=True):
    z = grid_coords_t[:, 2]
    weights = 1.0 / torch.pow(z + z0, beta)
    if normalize:
        weights = weights / (weights.mean() + 1e-12)
    return weights


def cg_solve(matvec, b, x0=None, max_iter=500, tol=1e-6):
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    bnorm = torch.sqrt(torch.dot(b, b) + 1e-30)
    for _ in range(max_iter):
        Ap = matvec(p)
        alpha = rs_old / (torch.dot(p, Ap) + 1e-30)
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        if float(torch.sqrt(rs_new) / bnorm) < tol:
            break
        p = r + (rs_new / (rs_old + 1e-30)) * p
        rs_old = rs_new
    return x


def weighted_mse(gz_pred, gz_obs, sigma):
    return float(torch.mean(((gz_pred - gz_obs) / sigma) ** 2).item())


def run_smooth_inversion(G, gz_obs, sigma, grid_coords, nx, ny, nz, dx, dy, dz, device, reg_scale=1.0):
    wd = 1.0 / sigma
    Gw = wd * G
    dw = wd * gz_obs
    GT = Gw.T.contiguous()
    b = GT @ dw

    grid_t = torch.tensor(grid_coords, dtype=torch.float32, device=device)
    w2 = depth_weights(grid_t, SMOOTH_CFG['z0'], SMOOTH_CFG['beta']) ** 2
    Dx, Dy, Dz = build_grad_ops_sparse(nx, ny, nz, dx, dy, dz, device)
    alpha_s = reg_scale * SMOOTH_CFG['alpha_s']
    alpha_x = reg_scale * SMOOTH_CFG['alpha_x']
    alpha_y = reg_scale * SMOOTH_CFG['alpha_y']
    alpha_z = reg_scale * SMOOTH_CFG['alpha_z']

    def matvec(m):
        out = GT @ (Gw @ m)
        out += alpha_s ** 2 * (w2 * m)
        for D, alpha in [(Dx, alpha_x), (Dy, alpha_y), (Dz, alpha_z)]:
            out += alpha ** 2 * torch.sparse.mm(D.T, torch.sparse.mm(D, m.unsqueeze(1))).squeeze(1)
        return out

    m_inv = cg_solve(matvec, b, max_iter=CG_MAX_ITER, tol=CG_TOL)
    gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)
    return m_inv, gz_pred, weighted_mse(gz_pred, gz_obs, sigma)


def run_tv_inversion(G, gz_obs, sigma, grid_coords, nx, ny, nz, dx, dy, dz, device, reg_scale=1.0):
    wd = 1.0 / sigma
    Gw = wd * G
    dw = wd * gz_obs
    GT = Gw.T.contiguous()
    b = GT @ dw

    grid_t = torch.tensor(grid_coords, dtype=torch.float32, device=device)
    w2 = depth_weights(grid_t, TV_CFG['z0'], TV_CFG['beta']) ** 2
    Dx, Dy, Dz = build_grad_ops_sparse(nx, ny, nz, dx, dy, dz, device)
    m_current = torch.zeros(nx * ny * nz, dtype=torch.float32, device=device)
    alpha_s = reg_scale * TV_CFG['alpha_s']
    alpha_tv = reg_scale * TV_CFG['alpha_tv']

    for _ in range(TV_CFG['irls_iters']):
        gx = torch.sparse.mm(Dx, m_current.unsqueeze(1)).squeeze(1)
        gy = torch.sparse.mm(Dy, m_current.unsqueeze(1)).squeeze(1)
        gz_grad = torch.sparse.mm(Dz, m_current.unsqueeze(1)).squeeze(1)
        wx = 1.0 / torch.sqrt(gx ** 2 + TV_CFG['irls_eps'] ** 2)
        wy = 1.0 / torch.sqrt(gy ** 2 + TV_CFG['irls_eps'] ** 2)
        wz = 1.0 / torch.sqrt(gz_grad ** 2 + TV_CFG['irls_eps'] ** 2)

        Wx = torch.sparse_coo_tensor(torch.stack([torch.arange(wx.shape[0], device=device)] * 2), wx, size=(wx.shape[0], wx.shape[0])).coalesce()
        Wy = torch.sparse_coo_tensor(torch.stack([torch.arange(wy.shape[0], device=device)] * 2), wy, size=(wy.shape[0], wy.shape[0])).coalesce()
        Wz = torch.sparse_coo_tensor(torch.stack([torch.arange(wz.shape[0], device=device)] * 2), wz, size=(wz.shape[0], wz.shape[0])).coalesce()

        def matvec(m):
            out = GT @ (Gw @ m)
            out += alpha_s ** 2 * (w2 * m)
            out += alpha_tv ** 2 * torch.sparse.mm(Dx.T, torch.sparse.mm(Wx, torch.sparse.mm(Dx, m.unsqueeze(1)))).squeeze(1)
            out += alpha_tv ** 2 * torch.sparse.mm(Dy.T, torch.sparse.mm(Wy, torch.sparse.mm(Dy, m.unsqueeze(1)))).squeeze(1)
            out += alpha_tv ** 2 * torch.sparse.mm(Dz.T, torch.sparse.mm(Wz, torch.sparse.mm(Dz, m.unsqueeze(1)))).squeeze(1)
            return out

        m_current = cg_solve(matvec, b, x0=m_current, max_iter=CG_MAX_ITER, tol=CG_TOL)

    gz_pred = (G @ m_current.unsqueeze(1)).squeeze(1)
    return m_current, gz_pred, weighted_mse(gz_pred, gz_obs, sigma)


def tune_explicit_inversion(label, solver, target, tol, *solver_args):
    lower_bound = target - tol
    upper_bound = target + tol
    trials = {}

    def evaluate(scale):
        scale = float(scale)
        if scale in trials:
            return trials[scale]
        model_vec, gz_pred, chi2 = solver(*solver_args, reg_scale=scale)
        result = {
            'scale': scale,
            'model': model_vec,
            'gz_pred': gz_pred,
            'weighted_mse': chi2,
        }
        trials[scale] = result
        print(f'{label}: reg_scale = {scale:.6g}, weighted MSE = {chi2:.4f}')
        return result

    current = evaluate(1.0)
    if lower_bound <= current['weighted_mse'] <= upper_bound:
        return current

    if current['weighted_mse'] < lower_bound:
        low = current
        high = None
        while low['weighted_mse'] < lower_bound and low['scale'] < REG_SCALE_MAX:
            high = evaluate(min(low['scale'] * 2.0, REG_SCALE_MAX))
            if high['weighted_mse'] >= lower_bound or high['scale'] >= REG_SCALE_MAX:
                break
            low = high
        if high is None:
            high = low
    else:
        high = current
        low = None
        while high['weighted_mse'] > upper_bound and high['scale'] > REG_SCALE_MIN:
            low = evaluate(max(high['scale'] / 2.0, REG_SCALE_MIN))
            if low['weighted_mse'] <= upper_bound or low['scale'] <= REG_SCALE_MIN:
                break
            high = low
        if low is None:
            low = high

    candidates = list(trials.values())
    if low['scale'] != high['scale']:
        left = min(low['scale'], high['scale'])
        right = max(low['scale'], high['scale'])
        for _ in range(REG_SEARCH_STEPS):
            mid = float(np.sqrt(left * right))
            mid_result = evaluate(mid)
            candidates.append(mid_result)
            if mid_result['weighted_mse'] < lower_bound:
                left = mid
            else:
                right = mid

    safe_candidates = [item for item in trials.values() if item['weighted_mse'] >= lower_bound]
    if safe_candidates:
        return min(safe_candidates, key=lambda item: abs(item['weighted_mse'] - target))
    return max(trials.values(), key=lambda item: item['weighted_mse'])


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_results(save_path, true_model, results, gz_obs, obs_points, x, y, z, dx, dz):
    nx, ny, nz = true_model.shape
    ix = nx // 2
    y_edge = [y[0] - dx / 2, y[-1] + dx / 2]
    z_edge = [z[0] - dz / 2, z[-1] + dz / 2]
    extent_yz = [y_edge[0], y_edge[1], z_edge[1], z_edge[0]]

    fig, axes = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)

    true_yz = true_model[ix, :, :].cpu().numpy()
    im = axes[0, 0].imshow(true_yz.T, origin='upper', extent=extent_yz, aspect='equal', vmin=0, vmax=RHO_BLK, cmap=CMAP)
    axes[0, 0].set_title(f'True YZ @ x~{x[ix]:.0f} m', fontsize=TITLE_FONTSIZE)
    style_axes(axes[0, 0], 'y (m)', 'Depth (m)')

    model_keys = ['inr', 'smooth', 'tv']
    model_titles = ['INR (implicit)', 'Smoothness', 'TV']
    for column, (key, title) in enumerate(zip(model_keys, model_titles), start=1):
        pred = results[key]['model'].view(nx, ny, nz).cpu().numpy()
        axes[0, column].imshow(pred[ix, :, :].T, origin='upper', extent=extent_yz, aspect='equal', vmin=0, vmax=RHO_BLK, cmap=CMAP)
        axes[0, column].set_title(title, fontsize=TITLE_FONTSIZE)
        style_axes(axes[0, column], 'y (m)', 'Depth (m)')

    fig.colorbar(im, ax=axes[0, :], label='kg/m3', fraction=0.025, pad=0.02, shrink=0.82)

    obs_x = obs_points[:, 0].cpu().numpy()
    obs_y = obs_points[:, 1].cpu().numpy()
    obs_mgal = MGAL_PER_MPS2 * gz_obs.detach().cpu().numpy()
    vmax_obs = np.abs(obs_mgal).max()
    sc = axes[1, 0].scatter(obs_x, obs_y, c=obs_mgal, s=60, cmap=CMAP, vmin=-vmax_obs, vmax=vmax_obs, edgecolors='none')
    axes[1, 0].set_title('Observed gz', fontsize=TITLE_FONTSIZE)
    style_axes(axes[1, 0], 'x (m)', 'y (m)')
    axes[1, 0].set_aspect('equal')

    residual_mappables = []
    for column, (key, title) in enumerate(zip(model_keys, model_titles), start=1):
        pred_mgal = MGAL_PER_MPS2 * results[key]['gz_pred'].detach().cpu().numpy()
        residual_mgal = obs_mgal - pred_mgal
        rms_res = np.sqrt(np.mean(residual_mgal ** 2))
        sc_res = axes[1, column].scatter(obs_x, obs_y, c=residual_mgal, s=60, cmap=CMAP,
                                         vmin=-np.abs(residual_mgal).max(), vmax=np.abs(residual_mgal).max(),
                                         edgecolors='none')
        axes[1, column].set_title(f'{title} residual\nRMS = {rms_res:.3f} mGal', fontsize=TITLE_FONTSIZE)
        style_axes(axes[1, column], 'x (m)', 'y (m)')
        axes[1, column].set_aspect('equal')
        residual_mappables.append(sc_res)

    fig.colorbar(sc, ax=[axes[1, 0]], label='mGal', fraction=0.046, pad=0.04, shrink=0.82)
    fig.colorbar(residual_mappables[-1], ax=[axes[1, 1], axes[1, 2], axes[1, 3]], label='mGal', fraction=0.025, pad=0.02, shrink=0.82)

    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def run():
    set_seed(DATA_SEED)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    x = np.arange(0.0, X_MAX + DX, DX)
    y = np.arange(0.0, Y_MAX + DY, DY)
    z = np.arange(0.0, Z_MAX + DZ, DZ)
    nx, ny, nz = len(x), len(y), len(z)

    x3, y3, z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=1)

    coords_mean = grid_coords.mean(axis=0, keepdims=True)
    coords_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = torch.tensor((grid_coords - coords_mean) / (coords_std + 1e-12), dtype=torch.float32, device=device)

    cell_grid = torch.tensor(np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), DZ / 2.0)]), dtype=torch.float32, device=device)
    xx_obs, yy_obs = np.meshgrid(x, y, indexing='ij')
    obs = torch.tensor(np.column_stack([xx_obs.ravel(), yy_obs.ravel(), -np.ones(xx_obs.size)]), dtype=torch.float32, device=device)

    print('Assembling sensitivity G ...')
    start = time.time()
    G = construct_sensitivity_matrix_G(cell_grid, obs, DX, DY, device)
    G = G.clone().detach().requires_grad_(False)
    print(f'G shape = {tuple(G.shape)}, time = {time.time() - start:.2f}s')

    rho_true, rho_true_3d = make_block_model(nx, ny, nz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true = rho_true.to(device)
    with torch.no_grad():
        gz_true = torch.matmul(G, rho_true.unsqueeze(1)).squeeze(1)
    sigma_noise = NOISE_LEVEL * gz_true.std()
    gz_obs = gz_true + sigma_noise * torch.randn_like(gz_true)
    wd = 1.0 / sigma_noise
    model_rng_state = capture_rng_state()

    cfg = dict(gamma=GAMMA, epochs=INR_EPOCHS, lr=INR_LR)
    cfg.update(
        use_early_stopping=USE_EARLY_STOPPING,
        early_stop_min_epochs=EARLY_STOP_MIN_EPOCHS,
        early_stop_patience=EARLY_STOP_PATIENCE,
        early_stop_target=EARLY_STOP_TARGET,
        early_stop_tol=EARLY_STOP_TOL,
        early_stop_overfit_patience=EARLY_STOP_OVERFIT_PATIENCE,
        progress_every=INR_PROGRESS_EVERY,
    )

    print('\nINR inversion (implicit regularization)')
    restore_rng_state(model_rng_state)
    inr_model = DensityContrastINR(hidden_sizes=INR_HIDDEN_SIZES, num_freqs=INR_NUM_FREQS, rho_abs_max=RHO_ABS_MAX).to(device)
    optimizer = torch.optim.Adam(inr_model.parameters(), lr=cfg['lr'])
    inr_history = train_inr(inr_model, optimizer, coords_norm, G, gz_obs, wd, cfg)
    inr_pred = inr_model(coords_norm).view(-1).detach()
    gz_inr = (G @ inr_pred.unsqueeze(1)).squeeze(1)
    inr_weighted_mse = weighted_mse(gz_inr, gz_obs, sigma_noise)

    print('\nClassical inversion with smoothness regularization')
    smooth_result = tune_explicit_inversion(
        'Smoothness',
        run_smooth_inversion,
        EARLY_STOP_TARGET,
        EARLY_STOP_TOL,
        G, gz_obs, sigma_noise, grid_coords, nx, ny, nz, DX, DY, DZ, device,
    )

    print('\nClassical inversion with TV regularization')
    tv_result = tune_explicit_inversion(
        'TV',
        run_tv_inversion,
        EARLY_STOP_TARGET,
        EARLY_STOP_TOL,
        G, gz_obs, sigma_noise, grid_coords, nx, ny, nz, DX, DY, DZ, device,
    )

    results = {
        'inr': {'model': inr_pred, 'gz_pred': gz_inr, 'history': inr_history, 'weighted_mse': inr_weighted_mse},
        'smooth': {'model': smooth_result['model'], 'gz_pred': smooth_result['gz_pred'], 'weighted_mse': smooth_result['weighted_mse'], 'reg_scale': smooth_result['scale']},
        'tv': {'model': tv_result['model'], 'gz_pred': tv_result['gz_pred'], 'weighted_mse': tv_result['weighted_mse'], 'reg_scale': tv_result['scale']},
    }

    voxel_params = nx * ny * nz
    inr_params = count_params(inr_model)
    print(f'\nVoxel parameters: {voxel_params:,}')
    print(f'INR parameters  : {inr_params:,}')

    summary_lines = [
        f'Voxel parameters: {voxel_params:,}',
        f'INR parameters  : {inr_params:,}',
    ]
    for key, label in [('inr', 'INR (implicit)'), ('smooth', 'Smoothness'), ('tv', 'TV')]:
        model_vec = results[key]['model']
        gz_pred = results[key]['gz_pred']
        rms_rho = float(torch.sqrt(torch.mean((model_vec - rho_true) ** 2)).item())
        rms_gz = float(torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item()) * MGAL_PER_MPS2
        chi2 = results[key]['weighted_mse']
        if 'reg_scale' in results[key]:
            line = f'{label:14s} | RMS rho = {rms_rho:7.2f} kg/m3 | RMS gz = {rms_gz:.4f} mGal | chi2 = {chi2:.4f} | reg_scale = {results[key]["reg_scale"]:.6g}'
        else:
            line = f'{label:14s} | RMS rho = {rms_rho:7.2f} kg/m3 | RMS gz = {rms_gz:.4f} mGal | chi2 = {chi2:.4f}'
        print(line)
        summary_lines.append(line)

    with open('plots/ImplicitVsExplicitRegularisation_metrics.txt', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(summary_lines))

    plot_results(
        save_path='plots/ImplicitVsExplicitRegularisation.png',
        true_model=rho_true_3d,
        results=results,
        gz_obs=gz_obs,
        obs_points=obs,
        x=x,
        y=y,
        z=z,
        dx=DX,
        dz=DZ,
    )


if __name__ == '__main__':
    run()