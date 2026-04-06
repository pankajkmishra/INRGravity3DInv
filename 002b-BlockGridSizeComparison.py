"""
007b-BlockGridSizeComparison.py
================================
Grid-size comparison for INR gravity inversion on the block model.

The goal is to show that, for a fixed INR architecture, refining the
voxel discretization increases the voxel-model parameter count while the
INR parameter count remains fixed.
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

# --- Domain ------------------------------------------------------------
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

# --- Fixed INR architecture -------------------------------------------
NUM_FREQS = 2
HIDDEN_SIZES = [48, 48, 24]

# --- Grid-size sweep ---------------------------------------------------
GRID_CONFIGS = {
    'C': 80.0,
    'M': 50.0,
    'F': 40.0,
}

GRID_LABELS = {
    'C': 'Coarse',
    'M': 'Medium',
    'F': 'Fine',
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


def make_block_model(x, y, z, rho_bg=0.0, rho_blk=400.0):
    x3, y3, z3 = np.meshgrid(x, y, z, indexing='ij')

    # Continuous version of the original stair-step block.
    inside_x = (x3 >= 325.0) & (x3 <= 625.0)
    inside_z = (z3 >= 25.0) & (z3 <= 375.0)
    inside_y = (y3 >= (575.0 - z3)) & (y3 <= (825.0 - z3))

    model = np.full(x3.shape, rho_bg, dtype=np.float32)
    model[inside_x & inside_y & inside_z] = rho_blk
    model_tensor = torch.from_numpy(model)
    return model_tensor.view(-1), model_tensor


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


def grid_label(label):
    return GRID_LABELS.get(label, label)


def solve_case(label, cell_size, device, cfg):
    set_seed(DATA_SEED)

    dx = dy = dz = cell_size
    x = np.arange(0.0, X_MAX + dx, dx)
    y = np.arange(0.0, Y_MAX + dy, dy)
    z = np.arange(0.0, Z_MAX + dz, dz)
    nx, ny, nz = len(x), len(y), len(z)

    x3, y3, z3 = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=1)

    coords_mean = grid_coords.mean(axis=0, keepdims=True)
    coords_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - coords_mean) / (coords_std + 1e-12)
    coords_norm = torch.tensor(coords_norm, dtype=torch.float32, device=device)

    cell_grid = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz / 2.0)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    xx_obs, yy_obs = np.meshgrid(x, y, indexing='ij')
    obs_points = np.column_stack([xx_obs.ravel(), yy_obs.ravel(), -np.ones(xx_obs.size)])
    obs = torch.tensor(obs_points, dtype=torch.float32, device=device)

    print(f'\nGrid {label}: dx = dy = dz = {cell_size:.1f} m')
    print('Assembling sensitivity G ...')
    start = time.time()
    G = construct_sensitivity_matrix_G(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)
    print(f'G shape = {tuple(G.shape)}, time = {time.time() - start:.2f}s')

    rho_true, rho_true_3d = make_block_model(x, y, z, rho_bg=RHO_BG, rho_blk=RHO_BLK)
    rho_true = rho_true.to(device)

    with torch.no_grad():
        gz_true = torch.matmul(G, rho_true.unsqueeze(1)).squeeze(1)
    sigma_noise = NOISE_LEVEL * gz_true.std()
    noise = sigma_noise * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    wd = 1.0 / sigma_noise

    model = DensityContrastINR(
        hidden_sizes=HIDDEN_SIZES,
        num_freqs=NUM_FREQS,
        rho_abs_max=RHO_ABS_MAX,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    history = train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg)
    rho_pred, gz_pred, rms_rho, rms_gz = evaluate_model(model, coords_norm, G, gz_obs, rho_true)

    return {
        'label': label,
        'cell_size': cell_size,
        'grid_shape': (nx, ny, nz),
        'voxel_params': nx * ny * nz,
        'inr_params': count_params(model),
        'history': history,
        'rho_true': rho_true.detach().cpu(),
        'rho_pred': rho_pred.detach().cpu(),
        'rms_rho': rms_rho,
        'rms_gz': rms_gz,
        'coords': (x, y, z),
    }


def plot_results(save_path, results):
    ordered_labels = list(GRID_CONFIGS.keys())
    fig, axes = plt.subplot_mosaic(
        [
            ['loss', 'loss', 'params', 'params'],
            ['true', 'C', 'M', 'F'],
        ],
        figsize=(18, 8),
        constrained_layout=True,
    )

    for label in ordered_labels:
        result = results[label]
        axes['loss'].plot(result['history']['loss'], linewidth=2.0, label=grid_label(label))
    axes['loss'].set_title('Training Loss', fontsize=TITLE_FONTSIZE)
    axes['loss'].set_yscale('log')
    axes['loss'].legend(fontsize=LEGEND_FONTSIZE)
    style_axes(axes['loss'], 'Epoch', 'Loss = mean((residual / sigma)^2)')

    voxel_params = [results[label]['voxel_params'] for label in ordered_labels]
    inr_params = [results[label]['inr_params'] for label in ordered_labels]

    display_labels = [grid_label(label) for label in ordered_labels]
    axes['params'].plot(display_labels, voxel_params, marker='o', linewidth=2.0, label='Voxel model', color='tab:blue')
    axes['params'].plot(display_labels, inr_params, marker='o', linewidth=2.0, label='INR model', color='tab:purple')
    for label, voxel_count, inr_count in zip(ordered_labels, voxel_params, inr_params):
        display_label = grid_label(label)
        axes['params'].annotate(f'{voxel_count:,}', (display_label, voxel_count), textcoords='offset points', xytext=(6, 6), fontsize=9, color='tab:blue')
        axes['params'].annotate(f'{inr_count:,}', (display_label, inr_count), textcoords='offset points', xytext=(6, -14), fontsize=9, color='tab:purple')
    axes['params'].set_title('Parameter Growth', fontsize=TITLE_FONTSIZE)
    axes['params'].set_yscale('log')
    axes['params'].legend(fontsize=LEGEND_FONTSIZE)
    style_axes(axes['params'], 'Grid size', 'Trainable parameters')

    finest_label = ordered_labels[-1]
    finest = results[finest_label]
    x, y, z = finest['coords']
    nx, ny, nz = finest['grid_shape']
    ix = nx // 2
    dy = y[1] - y[0] if len(y) > 1 else 1.0
    dz = z[1] - z[0] if len(z) > 1 else 1.0
    extent_true = [y[0] - dy / 2, y[-1] + dy / 2, z[-1] + dz / 2, z[0] - dz / 2]
    true_yz = finest['rho_true'].view(nx, ny, nz)[ix, :, :].numpy()

    im = axes['true'].imshow(true_yz.T, origin='upper', extent=extent_true, aspect='equal', vmin=0, vmax=RHO_BLK, cmap=CMAP)
    axes['true'].set_title(f'True YZ @ x~{x[ix]:.0f} m', fontsize=TITLE_FONTSIZE)
    style_axes(axes['true'], 'y (m)', 'Depth (m)')

    for label in ordered_labels:
        result = results[label]
        x, y, z = result['coords']
        nx, ny, nz = result['grid_shape']
        ix = nx // 2
        dy = y[1] - y[0] if len(y) > 1 else 1.0
        dz = z[1] - z[0] if len(z) > 1 else 1.0
        extent = [y[0] - dy / 2, y[-1] + dy / 2, z[-1] + dz / 2, z[0] - dz / 2]
        pred_yz = result['rho_pred'].view(nx, ny, nz)[ix, :, :].numpy()
        axes[label].imshow(pred_yz.T, origin='upper', extent=extent, aspect='equal', vmin=0, vmax=RHO_BLK, cmap=CMAP)
        axes[label].set_title(f"Recovered {grid_label(label)}", fontsize=TITLE_FONTSIZE)
        style_axes(axes[label], 'y (m)', 'Depth (m)')

    fig.colorbar(
        im,
        ax=[axes['true'], axes['C'], axes['M'], axes['F']],
        label='kg/m3',
        fraction=0.025,
        pad=0.02,
    )

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for label, cell_size in GRID_CONFIGS.items():
        results[label] = solve_case(label, cell_size, device, cfg)

    plot_results('plots/BlockGridSizeComparison.png', results)

    inr_params = results[next(iter(results))]['inr_params']
    print(f'\nFixed INR parameters: {inr_params:,}')
    for label in GRID_CONFIGS:
        result = results[label]
        nx, ny, nz = result['grid_shape']
        print(
            f"{grid_label(label)} | grid = {nx}x{ny}x{nz} | "
            f"voxel params = {result['voxel_params']:,} | "
            f"INR/voxel = {result['inr_params'] / result['voxel_params']:.3f} | "
            f"rho RMSE = {result['rms_rho']:.2f} kg/m^3 | "
            f"data RMSE = {result['rms_gz']:.3f} mGal | "
            f"best epoch = {result['history']['best_epoch']}"
        )


if __name__ == '__main__':
    run()