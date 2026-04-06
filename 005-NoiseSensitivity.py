"""
005-NoiseSensitivity.py
=======================
Noise-sensitivity study for block-model INR gravity inversion.

This script keeps the same block-model INR setting used in the other
blocky examples and varies only the data-noise model and level.
The observed data are changed across noise conditions, but the model
initialization is kept fixed so the comparison isolates robustness to
data contamination rather than optimizer randomness.
"""

import os
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


MGAL_PER_MPS2 = 1e5


# --- Random seeds ------------------------------------------------------
SEED = 42
DATA_SEED = SEED
MODEL_SEED_BASE = 1101

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

# --- Noise study -------------------------------------------------------
NOISE_TYPES = ('gaussian', 'correlated', 'outliers')
NOISE_LABELS = {
    'gaussian': 'Gaussian Noise',
    'correlated': 'Correlated Noise',
    'outliers': 'Outlier Noise',
}
NOISE_LEVELS = (0.02, 0.05, 0.10)
NOISE_LEVEL_LABELS = {
    0.02: 'Moderate (2%)',
    0.05: 'High (5%)',
    0.10: 'Severe (10%)',
}
OUTLIER_FRACTION = 0.05
OUTLIER_SCALE = 5.0

# --- INR training ------------------------------------------------------
INR_EPOCHS = 500
INR_LR = 1e-2
INR_NUM_FREQS = 2
INR_HIDDEN_SIZES = [256, 256, 256, 256]
GAMMA = 1.0
INR_PROGRESS_EVERY = 50

# --- Early stopping ----------------------------------------------------
USE_EARLY_STOPPING = True
EARLY_STOP_MIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
EARLY_STOP_TARGET = 1.0
EARLY_STOP_TOL = 0.05
EARLY_STOP_OVERFIT_PATIENCE = 5

# --- Plotting ----------------------------------------------------------
CMAP = 'Spectral_r'
FIG_DPI = 300
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
MODEL_VMAX = 250.0


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def a_integral_torch(x, y, z):
    eps = 1e-20
    radius = torch.sqrt(x ** 2 + y ** 2 + z ** 2).clamp_min(eps)
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


def make_gaussian_kernel(size=5, sigma=1.25, device='cpu'):
    coords = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing='ij')
    kernel = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, size, size)


def add_noise(gz_true, noise_type, noise_frac, obs_shape, noise_seed, device):
    set_seed(noise_seed)
    sigma = noise_frac * gz_true.std()
    n_obs = gz_true.numel()

    if noise_type == 'gaussian':
        noise = sigma * torch.randn(n_obs, device=device)

    elif noise_type == 'correlated':
        raw = torch.randn(obs_shape, dtype=torch.float32, device=device)
        kernel = make_gaussian_kernel(device=device)
        smoothed = F.conv2d(raw.view(1, 1, *obs_shape), kernel, padding=2).view(-1)
        smoothed = smoothed / (smoothed.std() + 1e-12)
        noise = sigma * smoothed

    elif noise_type == 'outliers':
        noise = sigma * torch.randn(n_obs, device=device)
        n_outliers = max(1, int(OUTLIER_FRACTION * n_obs))
        indices = torch.randperm(n_obs, device=device)[:n_outliers]
        noise[indices] += OUTLIER_SCALE * sigma * torch.randn(n_outliers, device=device)

    else:
        raise ValueError(f'Unsupported noise type: {noise_type}')

    return gz_true + noise, sigma


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


def weighted_mse(gz_pred, gz_obs, sigma):
    return float(torch.mean(((gz_pred - gz_obs) / sigma) ** 2).item())


def train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg, label):
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
        chi2 = float(loss.item() / cfg['gamma'])

        if progress_every and ((epoch + 1) == 1 or (epoch + 1) % progress_every == 0 or (epoch + 1) == cfg['epochs']):
            print(f'  {label} epoch {epoch + 1:4d}/{cfg["epochs"]}: chi2 = {chi2:.4f}')

        if use_early_stopping:
            gap = abs(chi2 - target)
            if gap < best_gap:
                best_gap = gap
                best_epoch = epoch
                best_weighted_mse = chi2
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if chi2 >= lower_bound and gap < best_safe_gap:
                best_safe_gap = gap
                best_safe_epoch = epoch
                best_safe_weighted_mse = chi2
                best_safe_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if epoch + 1 >= min_epochs:
                if target - tol <= chi2 <= target + tol:
                    in_band_count += 1
                else:
                    in_band_count = 0

                if chi2 < target - tol:
                    overfit_count += 1
                else:
                    overfit_count = 0

                if in_band_count >= patience or overfit_count >= overfit_patience:
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


def style_axes(ax, xlabel=None, ylabel=None):
    if xlabel is not None:
        ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    if ylabel is not None:
        ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_results(save_path, results, x, y, z, dy, dz):
    n_rows = len(NOISE_TYPES)
    n_cols = len(NOISE_LEVELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.5 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    z_edge = [z[0] - dz / 2, z[-1] + dz / 2]
    y_edge = [y[0] - dy / 2, y[-1] + dy / 2]
    extent_yz = [y_edge[0], y_edge[1], z_edge[1], z_edge[0]]

    color_image = None
    for row_index, noise_type in enumerate(NOISE_TYPES):
        for col_index, noise_level in enumerate(NOISE_LEVELS):
            ax = axes[row_index, col_index]
            yz_section = results[(noise_type, noise_level)]['model_yz']
            color_image = ax.imshow(
                yz_section.T,
                origin='upper',
                extent=extent_yz,
                aspect='equal',
                vmin=0.0,
                vmax=MODEL_VMAX,
                cmap=CMAP,
            )
            if row_index == 0:
                ax.set_title(NOISE_LEVEL_LABELS[noise_level], fontsize=TITLE_FONTSIZE)
            if col_index == 0:
                style_axes(ax, ylabel=f'{NOISE_LABELS[noise_type]}\nDepth (m)')
            else:
                style_axes(ax)
            if row_index == n_rows - 1:
                ax.set_xlabel('y (m)', fontsize=LABEL_FONTSIZE)

    fig.colorbar(color_image, ax=axes, label='kg/m3', fraction=0.025, pad=0.02, shrink=0.92)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_convergence(save_path, results):
    n_rows = len(NOISE_TYPES)
    n_cols = len(NOISE_LEVELS)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 3.2 * n_rows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for row_index, noise_type in enumerate(NOISE_TYPES):
        for col_index, noise_level in enumerate(NOISE_LEVELS):
            ax = axes[row_index, col_index]
            history = results[(noise_type, noise_level)]['history']
            epochs = np.arange(1, len(history) + 1)
            ax.plot(epochs, history, color='tab:blue', linewidth=1.8)
            ax.axhline(EARLY_STOP_TARGET, color='black', linestyle='--', linewidth=1.0)
            ax.set_yscale('log')
            if row_index == 0:
                ax.set_title(NOISE_LEVEL_LABELS[noise_level], fontsize=TITLE_FONTSIZE)
            if col_index == 0:
                style_axes(ax, ylabel=f'{NOISE_LABELS[noise_type]}\nChi2')
            else:
                style_axes(ax)
            if row_index == n_rows - 1:
                ax.set_xlabel('Epoch', fontsize=LABEL_FONTSIZE)

    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def run():
    device = torch.device('cpu')
    os.makedirs('plots', exist_ok=True)
    set_seed(DATA_SEED)

    x = np.arange(0.0, X_MAX + DX, DX)
    y = np.arange(0.0, Y_MAX + DY, DY)
    z = np.arange(0.0, Z_MAX + DZ, DZ)
    nx, ny, nz = len(x), len(y), len(z)

    x3, y3, z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=1)
    coords_mean = grid_coords.mean(axis=0, keepdims=True)
    coords_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = torch.tensor((grid_coords - coords_mean) / (coords_std + 1e-12), dtype=torch.float32, device=device)

    cell_grid = torch.tensor(
        np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), DZ / 2.0)]),
        dtype=torch.float32,
        device=device,
    )
    xx_obs, yy_obs = np.meshgrid(x, y, indexing='ij')
    obs = torch.tensor(
        np.column_stack([xx_obs.ravel(), yy_obs.ravel(), -np.ones(xx_obs.size)]),
        dtype=torch.float32,
        device=device,
    )

    print('Assembling sensitivity G ...')
    start = time.time()
    G = construct_sensitivity_matrix_G(cell_grid, obs, DX, DY, device)
    G = G.clone().detach().requires_grad_(False)
    print(f'G shape = {tuple(G.shape)}, time = {time.time() - start:.2f}s')

    rho_true_vec, rho_true_3d = make_block_model(nx, ny, nz, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true_vec = rho_true_vec.to(device)
    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)

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

    ix = nx // 2
    obs_shape = xx_obs.shape
    results = {}
    metrics_lines = [
        'noise_type,noise_level_pct,noise_seed,chi2,rms_rho_kgm3,rms_gz_mgal,best_epoch',
    ]

    for type_index, noise_type in enumerate(NOISE_TYPES):
        for level_index, noise_level in enumerate(NOISE_LEVELS):
            noise_seed = DATA_SEED + 100 * type_index + level_index
            gz_obs, sigma_noise = add_noise(gz_true, noise_type, noise_level, obs_shape, noise_seed, device)
            wd = 1.0 / sigma_noise
            label = f'{noise_type[:4]}-{noise_level * 100:.1f}%'

            print(f'\nRunning {NOISE_LABELS[noise_type]} noise at {noise_level * 100:.1f}%')
            print(f'  noise seed = {noise_seed}')

            set_seed(MODEL_SEED_BASE)
            model = DensityContrastINR(
                hidden_sizes=INR_HIDDEN_SIZES,
                num_freqs=INR_NUM_FREQS,
                rho_abs_max=RHO_ABS_MAX,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
            history = train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg, label=label)

            with torch.no_grad():
                rho_pred = model(coords_norm).view(-1)
                gz_pred = (G @ rho_pred.unsqueeze(1)).squeeze(1)

            chi2 = weighted_mse(gz_pred, gz_obs, sigma_noise)
            rms_rho = float(torch.sqrt(torch.mean((rho_pred - rho_true_vec) ** 2)).item())
            rms_gz = float(torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item()) * MGAL_PER_MPS2

            results[(noise_type, noise_level)] = {
                'model_yz': rho_pred.detach().cpu().numpy().reshape(nx, ny, nz)[ix, :, :],
                'chi2': chi2,
                'rms_rho': rms_rho,
                'rms_gz': rms_gz,
                'history': history['loss'],
            }
            metrics_lines.append(
                f'{noise_type},{noise_level * 100:.1f},{noise_seed},{chi2:.6f},{rms_rho:.3f},{rms_gz:.6f},{history["best_epoch"] + 1}'
            )

            print(f'  chi2 = {chi2:.4f} | RMS rho = {rms_rho:.2f} kg/m3 | RMS gz = {rms_gz:.4f} mGal')

    with open('plots/NoiseSensitivity_metrics.txt', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(metrics_lines))
    print('Saved plots/NoiseSensitivity_metrics.txt')

    plot_results(
        save_path='plots/NoiseSensitivity.png',
        results=results,
        x=x,
        y=y,
        z=z,
        dy=DY,
        dz=DZ,
    )

    plot_convergence(
        save_path='plots/NoiseSensitivityConvergence.png',
        results=results,
    )


if __name__ == '__main__':
    run()