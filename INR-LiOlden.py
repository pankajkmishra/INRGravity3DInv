import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import ndimage

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

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=8, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.register_buffer('freqs', 2.0 ** torch.arange(0, num_freqs))

    def forward(self, x):
        parts = [x] if self.include_input else []
        for f in self.freqs:
            parts += [torch.sin(f * x), torch.cos(f * x)]
        return torch.cat(parts, dim=-1)

class DensityContrastINR(nn.Module):
    def __init__(self, nfreq=8, hidden=256, depth=5, rho_abs_max=600.0):
        super().__init__()
        self.pe = PositionalEncoding(num_freqs=nfreq, include_input=True)
        in_dim = 3 * (1 + 2 * nfreq)
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

def li_oldenburg_weight(z_coords, z0, beta):
    z0 = max(float(z0), 1e-6)
    return (z0 / (z_coords + z0)) ** (beta / 2.0)

def depth_weighted_smoothness(m_vec, Nx, Ny, Nz, dx, dy, dz, w_depth, alpha_x, alpha_y, alpha_z):
    """Li-Oldenburg style depth weighting: W*D*m"""
    m = m_vec.view(Nx, Ny, Nz)
    w = w_depth.view(Nx, Ny, Nz)

    # centered differences
    Dxm = (m[1:, :, :] - m[:-1, :, :]) / dx
    Dym = (m[:, 1:, :] - m[:, :-1, :]) / dy
    Dzm = (m[:, :, 1:] - m[:, :, :-1]) / dz

    # Li–Oldenburg-style edge weights: geometric mean
    wx = torch.sqrt(w[1:, :, :] * w[:-1, :, :])
    wy = torch.sqrt(w[:, 1:, :] * w[:, :-1, :])
    wz = torch.sqrt(w[:, :, 1:] * w[:, :, :-1])

    reg = (alpha_x * (wx * Dxm).pow(2).mean() +
           alpha_y * (wy * Dym).pow(2).mean() +
           alpha_z * (wz * Dzm).pow(2).mean())
    return reg

def train_inr(model, opt, coords_norm, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, w_depth, cfg):
    history = {k: [] for k in ['total', 'gravity', 'smallness', 'smoothness']}
    rho_ref = torch.full((Nx * Ny * Nz,), cfg['rho_ref'], device=coords_norm.device)
    for ep in range(cfg['epochs']):
        opt.zero_grad()
        m_pred = model(coords_norm).view(-1)
        gz_pred = torch.matmul(G, m_pred.unsqueeze(1)).squeeze(1)
        residual = gz_pred - gz_obs
        data_term = cfg['gamma'] * torch.mean((Wd * residual)**2)
        small = cfg['alpha_s'] * torch.mean((w_depth * (m_pred - rho_ref)) ** 2)
        smooth = depth_weighted_smoothness(m_pred, Nx, Ny, Nz, dx, dy, dz, w_depth, cfg['alpha_x'], cfg['alpha_y'], cfg['alpha_z'])
        loss = data_term + small + smooth
        loss.backward()
        opt.step()
        history['gravity'].append(float(data_term.item()))
        history['smallness'].append(float(small.item()))
        history['smoothness'].append(float(smooth.item()))
        history['total'].append(float(loss.item()))
        if ep % 50 == 0 or ep == cfg['epochs'] - 1:
            print(f"Epoch {ep:4d} | data {history['gravity'][-1]:.3e} | "
                  f"small {history['smallness'][-1]:.3e} | smooth {history['smoothness'][-1]:.3e} | "
                  f"total {history['total'][-1]:.3e}")
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

def get_anomaly_contours(true_model, threshold=200.0):
    return true_model > threshold

def run():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)
    dx = dy = 50.0
    dz = 50.0
    x = np.arange(0.0, 1000.0 + dx, dx)
    y = np.arange(0.0, 1000.0 + dy, dy)
    z = np.arange(0.0, 500.0 + dz, dz)
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
    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0)
    rho_true_vec = rho_true_vec.to(device)
    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)
    nl = 0.01
    sigma = nl * gz_true.std()  # True noise standard deviation
    noise = sigma * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    Wd = 1.0 / sigma  # Data weighting based on noise level
    gz_mean, gz_std = gz_obs.mean(), gz_obs.std()
    z_coords = torch.tensor(grid_coords[:, 2], dtype=torch.float32, device=device)
    w_depth = li_oldenburg_weight(z_coords, z0=50.0, beta=2)
    cfg = dict(gamma=1.0, epochs=300, lr=1e-2, alpha_s=5e-2, alpha_x=1e-3, alpha_y=1e-3, alpha_z=1e-3, rho_ref=0.0)
    model = DensityContrastINR(nfreq=2, hidden=256, depth=4, rho_abs_max=600.0).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
    hist = train_inr(model, opt, coords_norm, G, gz_obs, Wd, Nx, Ny, Nz, dx, dy, dz, w_depth, cfg)
    with torch.no_grad():
        m_inv = model(coords_norm).view(-1)
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)
    def get_axes_coords():
        x1d = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
        y1d = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
        z1d = grid_coords[:, 2].reshape(Nx, Ny, Nz)[0, 0, :]
        return x1d, y1d, z1d
    x1d, y1d, z1d = get_axes_coords()
    
    block_boundaries = get_block_boundaries(Nx, Ny, Nz)
    
    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    tru = rho_true_3d.cpu().numpy()
    inv = m_inv.view(Nx, Ny, Nz).detach().cpu().numpy()
    
    # Adaptive scaling based on data
    tru_max = tru.max()
    inv_max = inv.max()
    
    # Combined figure: density slices (top 2 rows) + data maps (bottom row)
    fig1, axes = plt.subplots(3, 3, figsize=(16, 15))
    extent_xy = [x1d.min(), x1d.max(), y1d.min(), y1d.max()]
    extent_xz = [x1d.min(), x1d.max(), z1d.max(), z1d.min()]
    extent_yz = [y1d.min(), y1d.max(), z1d.max(), z1d.min()]
    
    # Top row: True density model
    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 0].set_title(f"True Δρ XY @ z≈{z1d[iz]:.0f} m")
    fig1.colorbar(im, ax=axes[0, 0], label='kg/m³', fraction=0.046, pad=0.04)
    im = axes[0, 1].imshow(tru[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 1].set_title(f"True Δρ XZ @ y≈{y1d[iy]:.0f} m")
    im = axes[0, 2].imshow(tru[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 2].set_title(f"True Δρ YZ @ x≈{x1d[ix]:.0f} m")
    im = axes[1, 0].imshow(inv[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 0].set_title(f"INR Δρ XY @ z≈{z1d[iz]:.0f} m")
    boundary_for_z = next((b for b in block_boundaries if b[4] == iz), None)
    if boundary_for_z:
        xs, xe, ys, ye, _ = boundary_for_z
        rect = plt.Rectangle((x[xs] - dx/2, y[ys] - dy/2), 
                             (xe - xs) * dx, 
                             (ye - ys) * dy,
                             edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 0].add_patch(rect)
    fig1.colorbar(im, ax=axes[1, 0], label='kg/m³', fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(inv[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 1].set_title(f"INR Δρ XZ @ y≈{y1d[iy]:.0f} m")
    z_indices_in_slice = []
    x_range = None
    for b in block_boundaries:
        xs, xe, ys, ye, z_idx = b
        if ys <= iy < ye:
            z_indices_in_slice.append(z_idx)
            if x_range is None:
                x_range = (xs, xe)
    if z_indices_in_slice and x_range:
        min_z_idx, max_z_idx = min(z_indices_in_slice), max(z_indices_in_slice)
        xs, xe = x_range
        rect = plt.Rectangle((x[xs] - dx/2, z[min_z_idx] - dz/2), 
                             (xe - xs) * dx, 
                             (max_z_idx - min_z_idx + 1) * dz,
                             edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 1].add_patch(rect)

    im = axes[1, 2].imshow(inv[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 2].set_title(f"INR Δρ YZ @ x≈{x1d[ix]:.0f} m")
    for xs, xe, ys, ye, z_idx in block_boundaries:
        if xs <= ix < xe:
            rect = plt.Rectangle((y[ys] - dy/2, z[z_idx] - dz/2), 
                                 (ye - ys) * dy, 
                                 dz,
                                 edgecolor='white', facecolor='none', linewidth=2)
            axes[1, 2].add_patch(rect)
    
    # Adjust aspect ratios for cross-sections
    # XZ and YZ plots: increase vertical axis by using larger aspect ratio
    for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
        ax.set_aspect(1.0)  # Make vertical axis same size as horizontal
    
    # XY plots: equal aspect ratio (1:1)
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_aspect(1.0)
    
    # Add axis labels
    # XY plots (left column)
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    
    # XZ plots (middle column)
    for ax in [axes[0, 1], axes[1, 1]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Depth (m)')
    
    # YZ plots (right column)
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.set_xlabel('y (m)')
        ax.set_ylabel('Depth (m)')
    
    # Add data maps in the third row
    def to_mgal(g):
        return 1e5 * g.detach().cpu().numpy()
    obs_mgal = to_mgal(gz_obs)
    pre_mgal = to_mgal(gz_pred)
    res_mgal = obs_mgal - pre_mgal
    
    # Get observation point coordinates
    obs_x = obs[:, 0].cpu().numpy()
    obs_y = obs[:, 1].cpu().numpy()
    
    v = max(abs(obs_mgal).max(), abs(pre_mgal).max())
    
    # Third row: Data maps with scatter plots
    sc = axes[2, 0].scatter(obs_x, obs_y, c=obs_mgal, s=80, cmap='viridis', vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 0].set_title('Observed gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 0], fraction=0.046, pad=0.04)

    sc = axes[2, 1].scatter(obs_x, obs_y, c=pre_mgal, s=80, cmap='viridis', vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 1].set_title('Predicted gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 1], fraction=0.046, pad=0.04)

    vmax_res = np.abs(res_mgal).max()
    sc = axes[2, 2].scatter(obs_x, obs_y, c=res_mgal, s=80, cmap='viridis', vmin=-vmax_res, vmax=vmax_res, marker='o', edgecolors='none')
    axes[2, 2].set_title(f'Residual (RMS={np.sqrt(np.mean(res_mgal**2)):.3f} mGal)')
    fig1.colorbar(sc, ax=axes[2, 2], fraction=0.046, pad=0.04)    # Set equal aspect for data maps
    for ax in [axes[2, 0], axes[2, 1], axes[2, 2]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')
    
    for ax in axes[0, 1:]:
        ax.set_ylabel('Depth (m)')
    for ax in axes.ravel():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel() or '')
    fig1.tight_layout()
    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    fig1.savefig('plots/INR-LiOlden3D.png', dpi=300)
    plt.close(fig1)
    
    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    # Loss curve is now included in the combined figure
    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(hist['gravity'], color='black')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    fig3.tight_layout()
    fig3.savefig('plots/INR-LiOlden3D_loss.png', dpi=300)
    plt.close(fig3)
    rms_rho = torch.sqrt(torch.mean((m_inv - rho_true_vec.to(device)) ** 2)).item()
    rms_gz = torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item() * 1e5
    print(f"RMS density-contrast error ≈ {rms_rho:.2f} kg/m^3")
    print(f"RMS data misfit ≈ {rms_gz:.3f} mGal")

if __name__ == '__main__':
    run()