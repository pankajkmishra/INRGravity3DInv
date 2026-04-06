"""
007-BlockNetworkSizeComparison.py
================================
Network-size comparison for INR gravity inversion on the block model.

The goal is to show how INR model capacity changes with network size,
while the number of trainable parameters remains different from a direct
voxel parameterization on the same grid.
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

# --- Training / optimisation -------------------------------------------
GAMMA = 1.0
EPOCHS = 500
LR = 1e-2

# --- Early stopping ----------------------------------------------------
USE_EARLY_STOPPING = True
EARLY_STOP_MIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
EARLY_STOP_TARGET = 1.0
EARLY_STOP_TOL = 0.05
EARLY_STOP_OVERFIT_PATIENCE = 5

# --- INR network -------------------------------------------------------
NUM_FREQS = 2
MODEL_CONFIGS = {
    'XS': [32, 16],
    'S': [48, 24],
    'M': [64, 32],
    'L': [48, 48, 24],
}

# --- Plotting -----------------------------------------------------------
CMAP = 'Spectral_r'
FIG_DPI = 300
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11
LEGEND_FONTSIZE = 10


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
    print(f'Seed = {seed}')


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
    in_band_count = 0
    overfit_count = 0

    start_time = time.time()
    for epoch in range(cfg['epochs']):
        optimizer.zero_grad()
        rho_pred = model(coords_norm).view(-1)
        gz_pred = torch.matmul(G, rho_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        loss = cfg['gamma'] * torch.mean((wd * residual) ** 2)
        loss.backward()
        optimizer.step()
        history['loss'].append(float(loss.item()))

        if use_early_stopping:
            weighted_mse = float(loss.item() / cfg['gamma'])
            gap = abs(weighted_mse - target)
            if gap < best_gap:
                best_gap = gap
                best_epoch = epoch
                best_weighted_mse = weighted_mse
                best_state = {
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
                    break

    if use_early_stopping and best_state is not None:
        model.load_state_dict(best_state)
        history['best_epoch'] = best_epoch
        history['best_weighted_mse'] = best_weighted_mse
    else:
        history['best_epoch'] = len(history['loss']) - 1
        history['best_weighted_mse'] = history['loss'][-1] / cfg['gamma']

    history['train_time_s'] = time.time() - start_time
    return history


@torch.no_grad()
def evaluate_model(model, coords_norm, G, gz_obs, rho_true):
    rho_pred = model(coords_norm).view(-1)
    gz_pred = torch.matmul(G, rho_pred.unsqueeze(1)).squeeze(1)
    rms_rho = float(torch.sqrt(torch.mean((rho_pred - rho_true) ** 2)).item())
    rms_gz = float(torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item()) * MGAL_PER_MPS2
    return rho_pred, gz_pred, rms_rho, rms_gz


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_results(save_path, grid_coords, rho_true, results, nx, ny, nz, voxel_params):
    labels_sorted = sorted(results.keys(), key=lambda label: results[label]['n_params'])
    smallest_label = labels_sorted[0]
    middle_label = labels_sorted[len(labels_sorted) // 2]
    largest_label = labels_sorted[-1]

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    label_to_color = {
        label: colors[index % len(colors)]
        for index, label in enumerate(labels_sorted)
    }

    fig, axes = plt.subplots(2, 4, figsize=(22, 10), constrained_layout=True)

    for label in labels_sorted:
        axes[0, 0].plot(results[label]['history']['loss'],
                        linewidth=2.0,
                        label=f"{label} ({results[label]['n_params']:,})",
                        color=label_to_color[label])
    axes[0, 0].set_title('Training Loss', fontsize=TITLE_FONTSIZE)
    axes[0, 0].set_yscale('log')
    axes[0, 0].grid(True, which='both', ls='--', alpha=0.3)
    axes[0, 0].legend(fontsize=LEGEND_FONTSIZE)
    style_axes(axes[0, 0], 'Epoch', 'Loss = mean((residual / sigma)^2)')

    for label in labels_sorted:
        axes[0, 1].scatter(results[label]['n_params'],
                           results[label]['rms_rho'],
                           s=120,
                           color=label_to_color[label],
                           edgecolors='k',
                           linewidths=1.0)
        axes[0, 1].annotate(label,
                            (results[label]['n_params'], results[label]['rms_rho']),
                            textcoords='offset points',
                            xytext=(6, 6),
                            fontsize=10)
    axes[0, 1].axvline(voxel_params, color='gray', linestyle='--', linewidth=1.5)
    axes[0, 1].text(voxel_params, axes[0, 1].get_ylim()[1], ' voxel model',
                    ha='left', va='top', fontsize=10, color='gray')
    axes[0, 1].set_xscale('log')
    axes[0, 1].set_title('Model Error vs Parameters', fontsize=TITLE_FONTSIZE)
    style_axes(axes[0, 1], 'INR parameters', 'Density RMSE (kg/m^3)')

    for label in labels_sorted:
        axes[0, 2].scatter(results[label]['n_params'],
                           results[label]['rms_gz'],
                           s=120,
                           color=label_to_color[label],
                           edgecolors='k',
                           linewidths=1.0)
        axes[0, 2].annotate(label,
                            (results[label]['n_params'], results[label]['rms_gz']),
                            textcoords='offset points',
                            xytext=(6, 6),
                            fontsize=10)
    axes[0, 2].axvline(voxel_params, color='gray', linestyle='--', linewidth=1.5)
    axes[0, 2].text(voxel_params, axes[0, 2].get_ylim()[1], ' voxel model',
                    ha='left', va='top', fontsize=10, color='gray')
    axes[0, 2].set_xscale('log')
    axes[0, 2].set_title('Data Misfit vs Parameters', fontsize=TITLE_FONTSIZE)
    style_axes(axes[0, 2], 'INR parameters', 'RMS data misfit (mGal)')

    ratios = [results[label]['n_params'] / voxel_params for label in labels_sorted]
    axes[0, 3].plot(labels_sorted, ratios, marker='o', linewidth=2.0, color='tab:purple')
    axes[0, 3].axhline(1.0, color='gray', linestyle='--', linewidth=1.5)
    axes[0, 3].set_title('INR / Voxel Parameters', fontsize=TITLE_FONTSIZE)
    style_axes(axes[0, 3], 'Model size', 'INR / voxel parameters')
    axes[0, 3].grid(True, ls='--', alpha=0.3)

    x1d = grid_coords[:, 0].reshape(nx, ny, nz)[:, 0, 0]
    y1d = grid_coords[:, 1].reshape(nx, ny, nz)[0, :, 0]
    z1d = grid_coords[:, 2].reshape(nx, ny, nz)[0, 0, :]
    ix = nx // 2

    y_edge_min, y_edge_max = y1d[0] - DY / 2, y1d[-1] + DY / 2
    z_edge_min, z_edge_max = z1d[0] - DZ / 2, z1d[-1] + DZ / 2
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]

    true_yz = rho_true.view(nx, ny, nz)[ix, :, :].cpu().numpy()
    small_yz = results[smallest_label]['rho_pred'].view(nx, ny, nz)[ix, :, :]
    middle_yz = results[middle_label]['rho_pred'].view(nx, ny, nz)[ix, :, :]
    large_yz = results[largest_label]['rho_pred'].view(nx, ny, nz)[ix, :, :]

    im = axes[1, 0].imshow(true_yz.T,
                           origin='upper',
                           extent=extent_yz,
                           aspect='equal',
                           vmin=0,
                           vmax=RHO_BLK,
                           cmap=CMAP)
    axes[1, 0].set_title(f'True YZ @ x~{x1d[ix]:.0f} m', fontsize=TITLE_FONTSIZE)
    style_axes(axes[1, 0], 'y (m)', 'Depth (m)')
    fig.colorbar(im, ax=axes[1, 0], label='kg/m3', fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(small_yz.T,
                           origin='upper',
                           extent=extent_yz,
                           aspect='equal',
                           vmin=0,
                           vmax=RHO_BLK,
                           cmap=CMAP)
    axes[1, 1].set_title(f'Recovered {smallest_label}', fontsize=TITLE_FONTSIZE)
    style_axes(axes[1, 1], 'y (m)', 'Depth (m)')
    fig.colorbar(im, ax=axes[1, 1], label='kg/m3', fraction=0.046, pad=0.04)

    im = axes[1, 2].imshow(middle_yz.T,
                           origin='upper',
                           extent=extent_yz,
                           aspect='equal',
                           vmin=0,
                           vmax=RHO_BLK,
                           cmap=CMAP)
    axes[1, 2].set_title(f'Recovered {middle_label}', fontsize=TITLE_FONTSIZE)
    style_axes(axes[1, 2], 'y (m)', 'Depth (m)')
    fig.colorbar(im, ax=axes[1, 2], label='kg/m3', fraction=0.046, pad=0.04)

    im = axes[1, 3].imshow(large_yz.T,
                           origin='upper',
                           extent=extent_yz,
                           aspect='equal',
                           vmin=0,
                           vmax=RHO_BLK,
                           cmap=CMAP)
    axes[1, 3].set_title(f'Recovered {largest_label}', fontsize=TITLE_FONTSIZE)
    style_axes(axes[1, 3], 'y (m)', 'Depth (m)')
    fig.colorbar(im, ax=axes[1, 3], label='kg/m3', fraction=0.046, pad=0.04)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
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
    coords_norm = (grid_coords - coords_mean) / (coords_std + 1e-12)
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32, device=device)

    dz_half = DZ / 2.0
    cell_grid = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    xx_obs, yy_obs = np.meshgrid(x, y, indexing='ij')
    obs_points = np.column_stack([xx_obs.ravel(), yy_obs.ravel(), -np.ones(xx_obs.size)])
    obs = torch.tensor(obs_points, dtype=torch.float32, device=device)

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
    noise = sigma_noise * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    wd = 1.0 / sigma_noise
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

    voxel_params = nx * ny * nz
    results = {}
    for label, hidden_sizes in MODEL_CONFIGS.items():
        restore_rng_state(model_rng_state)
        print(f'\nRunning model size {label}: hidden layers = {hidden_sizes}')

        model = DensityContrastINR(
            hidden_sizes=hidden_sizes,
            num_freqs=NUM_FREQS,
            rho_abs_max=RHO_ABS_MAX,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        history = train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg)
        rho_pred, gz_pred, rms_rho, rms_gz = evaluate_model(model, coords_norm, G, gz_obs, rho_true)

        results[label] = {
            'n_params': count_params(model),
            'history': history,
            'rho_pred': rho_pred.detach().cpu(),
            'gz_pred': gz_pred.detach().cpu(),
            'rms_rho': rms_rho,
            'rms_gz': rms_gz,
            'hidden_sizes': hidden_sizes,
        }

    plot_results(
        save_path='plots/BlockNetworkSizeComparison.png',
        grid_coords=grid_coords,
        rho_true=rho_true.detach().cpu(),
        results=results,
        nx=nx,
        ny=ny,
        nz=nz,
        voxel_params=voxel_params,
    )

    print(f'\nVoxel model parameters: {voxel_params:,}')
    for label in sorted(results.keys(), key=lambda key: results[key]['n_params']):
        item = results[label]
        ratio = item['n_params'] / voxel_params
        print(
            f"{label:>2} | layers = {item['hidden_sizes']} | "
            f"params = {item['n_params']:,} | "
            f"INR/voxel = {ratio:.3f} | "
            f"rho RMSE = {item['rms_rho']:.2f} kg/m^3 | "
            f"data RMSE = {item['rms_gz']:.3f} mGal | "
            f"best epoch = {item['history']['best_epoch']}"
        )


if __name__ == '__main__':
    run()