"""
004-ModelEnsambles.py
=====================
Ensemble uncertainty analysis for the same block-model INR setting used
in the 002 and 003 examples.

This script keeps the same domain, grid spacing, block model, noise
level, positional encoding, INR architecture, and noise-floor early
stopping used in the medium-grid comparisons. The ensemble differs only
through the random network initialisation of each member, while the
observed data are kept fixed across the ensemble.
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
MODEL_SEED_BASE = 1000

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
INR_HIDDEN_SIZES = [256, 256, 256, 256]
GAMMA = 1.0
INR_PROGRESS_EVERY = 25

# --- Early stopping ----------------------------------------------------
USE_EARLY_STOPPING = True
EARLY_STOP_MIN_EPOCHS = 100
EARLY_STOP_PATIENCE = 25
EARLY_STOP_TARGET = 1.0
EARLY_STOP_TOL = 0.05
EARLY_STOP_OVERFIT_PATIENCE = 5

# --- Ensemble ----------------------------------------------------------
N_ENSEMBLE = 20
MEMBER_SEED_OFFSET = 100
N_MEMBER_PANELS = 6

# --- Plotting ----------------------------------------------------------
CMAP = 'Spectral_r'
FIG_DPI = 300
TITLE_FONTSIZE = 14
LABEL_FONTSIZE = 12
TICK_FONTSIZE = 11


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def capture_rng_state():
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }


def restore_rng_state(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])


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


def weighted_mse(gz_pred, gz_obs, sigma):
    return float(torch.mean(((gz_pred - gz_obs) / sigma) ** 2).item())


def train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg, member_label='INR'):
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
        weighted = float(loss.item() / cfg['gamma'])

        if progress_every and ((epoch + 1) == 1 or (epoch + 1) % progress_every == 0 or (epoch + 1) == cfg['epochs']):
            print(f'  {member_label} epoch {epoch + 1:4d}/{cfg["epochs"]}: weighted MSE = {weighted:.4f}')

        if use_early_stopping:
            gap = abs(weighted - target)
            if gap < best_gap:
                best_gap = gap
                best_epoch = epoch
                best_weighted_mse = weighted
                best_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if weighted >= lower_bound and gap < best_safe_gap:
                best_safe_gap = gap
                best_safe_epoch = epoch
                best_safe_weighted_mse = weighted
                best_safe_state = {
                    name: value.detach().cpu().clone()
                    for name, value in model.state_dict().items()
                }

            if epoch + 1 >= min_epochs:
                if target - tol <= weighted <= target + tol:
                    in_band_count += 1
                else:
                    in_band_count = 0

                if weighted < target - tol:
                    overfit_count += 1
                else:
                    overfit_count = 0

                if in_band_count >= patience or overfit_count >= overfit_patience:
                    print(f'  {member_label} early stop at epoch {epoch + 1}: weighted MSE = {weighted:.4f}')
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


def style_axes(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel, fontsize=LABEL_FONTSIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONTSIZE)
    ax.tick_params(labelsize=TICK_FONTSIZE)


def plot_ensemble_results(save_path, true_model, ensemble_mean, ensemble_std,
                          gz_obs, gz_pred_mean, obs_points, x, y, z, dx, dz):
    nx, ny, nz = true_model.shape
    ix = nx // 2
    y_edge = [y[0] - dx / 2, y[-1] + dx / 2]
    z_edge = [z[0] - dz / 2, z[-1] + dz / 2]
    extent_yz = [y_edge[0], y_edge[1], z_edge[1], z_edge[0]]

    mosaic = [
        ['true', 'true', 'mean', 'mean', 'std', 'std'],
        ['obs', 'obs', 'pred', 'pred', 'res', 'res'],
    ]
    fig, axes = plt.subplot_mosaic(mosaic, figsize=(16, 8.8), constrained_layout=True)

    true_yz = true_model[ix, :, :]
    mean_yz = ensemble_mean[ix, :, :]
    std_yz = ensemble_std[ix, :, :]
    model_vmax = 250.0

    im_model = axes['true'].imshow(true_yz.T, origin='upper', extent=extent_yz,
                                   aspect='equal', vmin=0.0, vmax=model_vmax, cmap=CMAP)
    axes['true'].set_title(f'True YZ @ x~{x[ix]:.0f} m', fontsize=TITLE_FONTSIZE)
    style_axes(axes['true'], 'y (m)', 'Depth (m)')

    im_mean = axes['mean'].imshow(mean_yz.T, origin='upper', extent=extent_yz,
                                  aspect='equal', vmin=0.0, vmax=model_vmax, cmap=CMAP)
    axes['mean'].set_title('Ensemble Mean', fontsize=TITLE_FONTSIZE)
    style_axes(axes['mean'], 'y (m)', 'Depth (m)')

    im_std = axes['std'].imshow(std_yz.T, origin='upper', extent=extent_yz,
                                aspect='equal', vmin=0.0, vmax=max(float(ensemble_std.max()), 1.0), cmap=CMAP)
    axes['std'].set_title('Ensemble Std', fontsize=TITLE_FONTSIZE)
    style_axes(axes['std'], 'y (m)', 'Depth (m)')

    fig.colorbar(im_model, ax=[axes['true'], axes['mean']], label='kg/m3', fraction=0.025, pad=0.02, shrink=0.82)
    fig.colorbar(im_std, ax=[axes['std']], label='kg/m3', fraction=0.046, pad=0.04, shrink=0.82)

    obs_x = obs_points[:, 0].cpu().numpy()
    obs_y = obs_points[:, 1].cpu().numpy()
    obs_mgal = MGAL_PER_MPS2 * gz_obs.detach().cpu().numpy()
    pred_mgal = MGAL_PER_MPS2 * gz_pred_mean.detach().cpu().numpy()
    residual_mgal = obs_mgal - pred_mgal
    vmax_grav = max(np.abs(obs_mgal).max(), np.abs(pred_mgal).max())
    vmax_res = np.abs(residual_mgal).max()

    sc_obs = axes['obs'].scatter(obs_x, obs_y, c=obs_mgal, s=60, cmap=CMAP,
                                vmin=-vmax_grav, vmax=vmax_grav, edgecolors='none')
    axes['obs'].set_title('Observed gz', fontsize=TITLE_FONTSIZE)
    style_axes(axes['obs'], 'x (m)', 'y (m)')
    axes['obs'].set_aspect('equal')

    sc_pred = axes['pred'].scatter(obs_x, obs_y, c=pred_mgal, s=60, cmap=CMAP,
                                 vmin=-vmax_grav, vmax=vmax_grav, edgecolors='none')
    axes['pred'].set_title('Ensemble Mean gz', fontsize=TITLE_FONTSIZE)
    style_axes(axes['pred'], 'x (m)', 'y (m)')
    axes['pred'].set_aspect('equal')

    sc_res = axes['res'].scatter(obs_x, obs_y, c=residual_mgal, s=60, cmap=CMAP,
                                vmin=-vmax_res, vmax=vmax_res, edgecolors='none')
    axes['res'].set_title(f'Residual\nRMS = {np.sqrt(np.mean(residual_mgal ** 2)):.3f} mGal', fontsize=TITLE_FONTSIZE)
    style_axes(axes['res'], 'x (m)', 'y (m)')
    axes['res'].set_aspect('equal')

    fig.colorbar(sc_obs, ax=[axes['obs'], axes['pred']], label='mGal', fraction=0.025, pad=0.02, shrink=0.82)
    fig.colorbar(sc_res, ax=[axes['res']], label='mGal', fraction=0.046, pad=0.04, shrink=0.82)

    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_member_metrics(save_path, member_stats):
    member_ids = np.arange(len(member_stats))
    chi2 = np.array([item['weighted_mse'] for item in member_stats])
    rms_values = np.array([item['rms_rho'] for item in member_stats])

    fig, ax_loss = plt.subplots(1, 1, figsize=(12, 4.2), constrained_layout=True)
    ax_rms = ax_loss.twinx()

    ax_loss.axhspan(EARLY_STOP_TARGET - EARLY_STOP_TOL, EARLY_STOP_TARGET + EARLY_STOP_TOL, color='0.92', zorder=0)
    ax_loss.axhline(EARLY_STOP_TARGET, color='black', linestyle='--', linewidth=1.0)
    ax_loss.plot(member_ids, chi2, linestyle='none', marker='o', color='tab:blue', markersize=8)
    ax_loss.set_yscale('log')
    ax_loss.set_title('Ensemble Member Metrics', fontsize=TITLE_FONTSIZE)
    style_axes(ax_loss, 'Member', 'Chi2')

    ax_rms.plot(member_ids, rms_values, linestyle='none', marker='s', color='tab:green', markersize=8)
    ax_rms.set_ylabel('Model Error (kg/m3)', fontsize=LABEL_FONTSIZE)
    ax_rms.tick_params(labelsize=TICK_FONTSIZE)

    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def plot_member_sections(save_path, ensemble_models, member_numbers, x, y, z, dx, dz):
    n_members, nx, ny, nz = ensemble_models.shape
    ix = nx // 2
    y_edge = [y[0] - dx / 2, y[-1] + dx / 2]
    z_edge = [z[0] - dz / 2, z[-1] + dz / 2]
    extent_yz = [y_edge[0], y_edge[1], z_edge[1], z_edge[0]]

    n_show = min(N_MEMBER_PANELS, n_members)
    ncols = 3
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 3.8 * nrows), constrained_layout=True)
    axes = np.atleast_2d(axes)

    for index in range(n_show):
        ax = axes[index // ncols, index % ncols]
        ax.imshow(ensemble_models[index, ix, :, :].T, origin='upper', extent=extent_yz,
                  aspect='equal', vmin=0.0, vmax=RHO_BLK, cmap=CMAP)
        ax.set_title(f'Member {member_numbers[index]:02d}', fontsize=TITLE_FONTSIZE)
        style_axes(ax, 'y (m)', 'Depth (m)')

    for index in range(n_show, nrows * ncols):
        axes[index // ncols, index % ncols].set_visible(False)

    fig.savefig(save_path, dpi=FIG_DPI)
    plt.close(fig)
    print(f'Saved {save_path}')


def run():
    set_seed(DATA_SEED)
    device = torch.device('cpu')
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

    print(f'\nTraining ensemble with {N_ENSEMBLE} members')
    ensemble_models = []
    ensemble_gz = []
    member_stats = []

    for member_index in range(N_ENSEMBLE):
        print(f'\nMember {member_index + 1}/{N_ENSEMBLE}')
        restore_rng_state(model_rng_state)
        set_seed(MODEL_SEED_BASE + MEMBER_SEED_OFFSET + member_index)

        model = DensityContrastINR(hidden_sizes=INR_HIDDEN_SIZES, num_freqs=INR_NUM_FREQS, rho_abs_max=RHO_ABS_MAX).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
        history = train_inr(model, optimizer, coords_norm, G, gz_obs, wd, cfg, member_label=f'M{member_index + 1:02d}')

        with torch.no_grad():
            rho_pred = model(coords_norm).view(-1).detach()
            gz_pred = (G @ rho_pred.unsqueeze(1)).squeeze(1)

        chi2 = weighted_mse(gz_pred, gz_obs, sigma_noise)
        rms_rho = float(torch.sqrt(torch.mean((rho_pred - rho_true) ** 2)).item())
        ensemble_models.append(rho_pred.cpu().numpy())
        ensemble_gz.append(gz_pred.cpu().numpy())
        member_stats.append({
            'member': member_index + 1,
            'weighted_mse': chi2,
            'rms_rho': rms_rho,
            'best_epoch': history['best_epoch'] + 1,
        })
        print(f'  Member {member_index + 1:02d} summary: chi2 = {chi2:.4f}, RMS rho = {rms_rho:.2f} kg/m3')

    ensemble_models = np.stack(ensemble_models, axis=0).reshape(N_ENSEMBLE, nx, ny, nz)
    ensemble_gz = np.stack(ensemble_gz, axis=0)
    ensemble_mean = ensemble_models.mean(axis=0)
    ensemble_median = np.median(ensemble_models, axis=0)
    ensemble_std = ensemble_models.std(axis=0, ddof=1)
    gz_pred_mean = torch.tensor(ensemble_gz.mean(axis=0), dtype=torch.float32, device=device)
    gz_pred_median = torch.tensor(np.median(ensemble_gz, axis=0), dtype=torch.float32, device=device)

    voxel_params = nx * ny * nz
    inr_params = count_params(model)
    mean_model = torch.tensor(ensemble_mean.reshape(-1), dtype=torch.float32, device=device)
    median_model = torch.tensor(ensemble_median.reshape(-1), dtype=torch.float32, device=device)
    rms_rho_mean = float(torch.sqrt(torch.mean((mean_model - rho_true) ** 2)).item())
    rms_rho_median = float(torch.sqrt(torch.mean((median_model - rho_true) ** 2)).item())
    chi2_mean = weighted_mse(gz_pred_mean, gz_obs, sigma_noise)
    chi2_median = weighted_mse(gz_pred_median, gz_obs, sigma_noise)

    print(f'\nVoxel parameters: {voxel_params:,}')
    print(f'INR parameters  : {inr_params:,}')
    print(f'Ensemble mean RMS rho = {rms_rho_mean:.2f} kg/m3')
    print(f'Ensemble mean chi2    = {chi2_mean:.4f}')
    print(f'Ensemble median RMS rho = {rms_rho_median:.2f} kg/m3')
    print(f'Ensemble median chi2    = {chi2_median:.4f}')
    print(f'Mean member chi2      = {np.mean([item["weighted_mse"] for item in member_stats]):.4f}')

    summary_lines = [
        f'Voxel parameters: {voxel_params:,}',
        f'INR parameters  : {inr_params:,}',
        f'Ensemble mean RMS rho = {rms_rho_mean:.2f} kg/m3',
        f'Ensemble mean chi2    = {chi2_mean:.4f}',
        f'Ensemble median RMS rho = {rms_rho_median:.2f} kg/m3',
        f'Ensemble median chi2    = {chi2_median:.4f}',
        f'Mean member chi2      = {np.mean([item["weighted_mse"] for item in member_stats]):.4f}',
        '',
        'Per-member summary:',
    ]
    for item in member_stats:
        summary_lines.append(
            f'Member {item["member"]:02d} | chi2 = {item["weighted_mse"]:.4f} | RMS rho = {item["rms_rho"]:.2f} kg/m3 | best epoch = {item["best_epoch"]}'
        )

    with open('plots/ModelEnsemble_metrics.txt', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(summary_lines))

    plot_ensemble_results(
        save_path='plots/ModelEnsembleMean.png',
        true_model=rho_true_3d.cpu().numpy(),
        ensemble_mean=ensemble_mean,
        ensemble_std=ensemble_std,
        gz_obs=gz_obs,
        gz_pred_mean=gz_pred_mean,
        obs_points=obs,
        x=x,
        y=y,
        z=z,
        dx=DX,
        dz=DZ,
    )

    plot_member_metrics(
        save_path='plots/ModelEnsembleMetrics.png',
        member_stats=member_stats,
    )

    plot_member_sections(
        save_path='plots/ModelEnsemble_members.png',
        ensemble_models=ensemble_models,
        member_numbers=[item['member'] for item in member_stats],
        x=x,
        y=y,
        z=z,
        dx=DX,
        dz=DZ,
    )


if __name__ == '__main__':
    run()
