import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random

MGAL_PER_MPS2 = 1e5
RHO_GC_TO_KGM3 = 1000.0

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

def A_integral_torch(x, y, z):
    Gamma = 6.67430e-11
    eps = 1e-12
    r = torch.sqrt(x**2 + y**2 + z**2)
    r_safe, z_safe = torch.clamp(r, min=eps), torch.clamp(z, min=eps)
    log_y_r, log_x_r = torch.log(torch.abs(y + r_safe)), torch.log(torch.abs(x + r_safe))
    arctan_term = torch.arctan((x * y) / (z_safe * r_safe))
    return -Gamma * (x * log_y_r + y * log_x_r - z * arctan_term)

def construct_sensitivity_matrix_G_torch(cell_grid, data_points, d1, d2, device):
    cell_x = cell_grid[:, 0].unsqueeze(0)
    cell_y = cell_grid[:, 1].unsqueeze(0)
    cell_z = cell_grid[:, 2].unsqueeze(0)
    cell_dz_half = cell_grid[:, 3].unsqueeze(0)
    obs_x = data_points[:, 0].unsqueeze(1)
    obs_y = data_points[:, 1].unsqueeze(1)
    obs_z = data_points[:, 2].unsqueeze(1)
    x2, x1 = (cell_x + d1 / 2) - obs_x, (cell_x - d1 / 2) - obs_x
    y2, y1 = (cell_y + d2 / 2) - obs_y, (cell_y - d2 / 2) - obs_y
    z2, z1 = cell_z + cell_dz_half - obs_z, cell_z - cell_dz_half - obs_z
    A = (A_integral_torch(x2, y2, z2) - A_integral_torch(x2, y2, z1) -
         A_integral_torch(x2, y1, z2) + A_integral_torch(x2, y1, z1) -
         A_integral_torch(x1, y2, z2) + A_integral_torch(x1, y2, z1) +
         A_integral_torch(x1, y1, z2) - A_integral_torch(x1, y1, z1))
    return A

def generate_grf_torch(nx, ny, nz, dx, dy, dz, lambda_val, nu, sigma, device):
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * torch.pi
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = Kx**2 + Ky**2 + Kz**2
    power_spectrum = (k_squared + (1 / lambda_val**2))**(-nu - 1.5)
    power_spectrum[0, 0, 0] = 0
    noise = torch.randn(nx, ny, nz, dtype=torch.complex64, device=device)
    fourier_field = noise * torch.sqrt(power_spectrum)
    real_field = torch.real(torch.fft.ifftn(fourier_field))
    real_field -= torch.mean(real_field)
    std_dev = torch.std(real_field)
    if std_dev > 1e-9:
        real_field /= std_dev
    real_field *= sigma
    return real_field

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.include_input = include_input
        freqs = 2.0 ** torch.arange(0, num_freqs, dtype=torch.float32)
        self.register_buffer("freqs", freqs, persistent=False)
    def forward(self, x):
        encoded = []
        if self.include_input:
            encoded.append(x)
        for f in self.freqs:
            encoded.append(torch.sin(x * f))
            encoded.append(torch.cos(x * f))
        return torch.cat(encoded, dim=-1)

def create_mlp(input_dim, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
        nn.Linear(256, 128), nn.LeakyReLU(0.01),
        nn.Linear(128, 64),  nn.LeakyReLU(0.01),
        nn.Linear(64, output_dim),
        nn.Sigmoid()
    )

class DensityModel(nn.Module):
    def __init__(self, use_posenc=True, num_freqs=10, min_density=1.6, max_density=3.5):
        super().__init__()
        self.use_posenc = use_posenc
        self.min_density = float(min_density)
        self.max_density = float(max_density)
        if use_posenc:
            self.positional_encoding = PositionalEncoding(num_freqs=num_freqs)
            input_dim = 3 * (1 + 2 * num_freqs)
        else:
            self.positional_encoding = None
            input_dim = 3
        self.density_net = create_mlp(input_dim, output_dim=1)
    def forward(self, coords):
        feats = self.positional_encoding(coords) if self.use_posenc else coords
        norm_density = self.density_net(feats)
        return self.min_density + norm_density * (self.max_density - self.min_density)

def train(model, optimizer, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor, Nx, Ny, Nz, epochs, gamma=1.0):
    model.train()
    history = {"total": [], "gravity": []}
    for _ in range(epochs):
        optimizer.zero_grad()
        rho_pred_gcc = model(coords_tensor).view(Nx * Ny * Nz)
        rho_pred_kgm3 = rho_pred_gcc * RHO_GC_TO_KGM3
        gz_pred_mps2 = G_tensor @ rho_pred_kgm3.unsqueeze(1)
        res_w = (gz_pred_mps2 - gz_obs_mps2) / sigma_mps2
        gravity_loss = gamma * F.mse_loss(res_w, torch.zeros_like(res_w))
        gravity_loss.backward()
        optimizer.step()
        history["gravity"].append(gravity_loss.item())
        history["total"].append(gravity_loss.item())
    return history

@torch.no_grad()
def evaluate_model(model, coords_tensor, G_tensor, Nx, Ny, Nz):
    model.eval()
    rho_pred_gcc = model(coords_tensor).flatten()
    rho_pred_kgm3 = rho_pred_gcc * RHO_GC_TO_KGM3
    gz_pred_mps2 = G_tensor @ rho_pred_kgm3.unsqueeze(1)
    return rho_pred_gcc.cpu(), gz_pred_mps2.cpu()

def plot_two_row_summary(rho_true_flat_gcc, rho_plain_gcc, rho_pos_gcc, gz_obs_mps2, gz_pred_plain_mps2, gz_pred_pos_mps2, hist_plain, hist_pos, grid_coords, obs_points, Nx, Ny, Nz, save_path, z_index=None):
    z_index = Nz // 2 if z_index is None else z_index
    rho_true_slice  = rho_true_flat_gcc.view(Nx, Ny, Nz)[:, :, z_index].cpu().numpy()
    rho_plain_slice = rho_plain_gcc.view(Nx, Ny, Nz)[:, :, z_index].cpu().numpy()
    rho_pos_slice   = rho_pos_gcc.view(Nx, Ny, Nz)[:, :, z_index].cpu().numpy()
    vmin_rho, vmax_rho = 1.6, 3.5
    x_coords = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y_coords = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    im0 = axes[0, 0].imshow(rho_true_slice.T, origin='lower', vmin=vmin_rho, vmax=vmax_rho, extent=extent, cmap='viridis', aspect='equal')
    axes[0, 0].set_title(f"True Density (z={z_index})")
    axes[0, 0].set_xlabel("x (m)")
    axes[0, 0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[0, 0], label='Density (g/cm³)')
    im1 = axes[0, 1].imshow(rho_plain_slice.T, origin='lower', vmin=vmin_rho, vmax=vmax_rho, extent=extent, cmap='viridis', aspect='equal')
    axes[0, 1].set_title("Inverted Density – Pure MLP")
    axes[0, 1].set_xlabel("x (m)")
    axes[0, 1].set_ylabel("")
    fig.colorbar(im1, ax=axes[0, 1], label='Density (g/cm³)')
    im2 = axes[0, 2].imshow(rho_pos_slice.T, origin='lower', vmin=vmin_rho, vmax=vmax_rho, extent=extent, cmap='viridis', aspect='equal')
    axes[0, 2].set_title("Inverted Density – PosEnc MLP")
    axes[0, 2].set_xlabel("x (m)")
    axes[0, 2].set_ylabel("")
    fig.colorbar(im2, ax=axes[0, 2], label='Density (g/cm³)')
    obs = np.asarray(gz_obs_mps2).ravel()
    pred_plain = np.asarray(gz_pred_plain_mps2).ravel()
    pred_pos   = np.asarray(gz_pred_pos_mps2).ravel()
    res_plain_mgal = (obs - pred_plain) * MGAL_PER_MPS2
    res_pos_mgal   = (obs - pred_pos) * MGAL_PER_MPS2
    lim = float(np.max(np.abs([res_plain_mgal.max(), res_plain_mgal.min(), res_pos_mgal.max(), res_pos_mgal.min()])))
    x_obs = obs_points[:, 0]
    y_obs = obs_points[:, 1]
    ax = axes[1, 0]
    ax.plot(hist_plain['gravity'], label='Pure MLP')
    ax.plot(hist_pos['gravity'], label='PosEnc MLP')
    ax.set_title("Loss comparison")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend()
    ax = axes[1, 1]
    sc1 = ax.scatter(x_obs, y_obs, c=res_plain_mgal, s=12, cmap='viridis', vmin=-lim, vmax=lim)
    ax.set_title("data misfit (Pure MLP)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc1, ax=ax, label="Gravity Residual (mGal)")
    ax = axes[1, 2]
    sc2 = ax.scatter(x_obs, y_obs, c=res_pos_mgal, s=12, cmap='viridis', vmin=-lim, vmax=lim)
    ax.set_title("data misfit (PosEnc MLP)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_aspect('equal', adjustable='box')
    fig.colorbar(sc2, ax=ax, label="Gravity Residual (mGal)")
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    config = {
        "gamma": 1.0,
        "epochs": 500,
        "lr": 0.001,
        "noise_level": 0.01,
        "seed": 41,
    }
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config["seed"])
    Nx, Ny, Nz = 40, 40, 20
    dx, dy, dz = 500.0, 500.0, 500.0
    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    z = np.linspace(0, (Nz - 1) * dz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    coords_mean = grid_coords.mean(axis=0, keepdims=True)
    coords_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - coords_mean) / coords_std
    coords_tensor = torch.tensor(coords_norm, dtype=torch.float32, device=device, requires_grad=True)
    dz_half = dx / 2.0
    cell_grid_np = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid_tensor = torch.tensor(cell_grid_np, dtype=torch.float32, device=device)
    X_obs, Y_obs = np.meshgrid(x, y, indexing='ij')
    obs_points_np = np.column_stack([X_obs.ravel(), Y_obs.ravel(), np.zeros_like(X_obs.ravel())])
    obs_points_tensor = torch.tensor(obs_points_np, dtype=torch.float32, device=device)
    rho_true_3d_gcc = generate_grf_torch(Nx, Ny, Nz, dx, dy, dz, 5000.0, 1.5, 2.0, device)
    min_val, max_val = torch.min(rho_true_3d_gcc), torch.max(rho_true_3d_gcc)
    rho_true_3d_gcc = 1.6 + (rho_true_3d_gcc - min_val) * ((3.5 - 1.6) / (max_val - min_val))
    rho_true_flat_gcc = rho_true_3d_gcc.flatten()
    G_tensor = construct_sensitivity_matrix_G_torch(cell_grid_tensor, obs_points_tensor, dx, dy, device)
    with torch.no_grad():
        gz_true_mps2 = G_tensor @ (rho_true_flat_gcc * RHO_GC_TO_KGM3).unsqueeze(1)
        nl = config["noise_level"]
        sigma_mps2 = nl * gz_true_mps2.std()
        noise = sigma_mps2 * torch.randn_like(gz_true_mps2)
        gz_obs_mps2 = gz_true_mps2 + noise
        _ = 1.0 / sigma_mps2
    model_plain = DensityModel(use_posenc=False).to(device)
    optim_plain = torch.optim.Adam(model_plain.parameters(), lr=config["lr"])
    hist_plain = train(model_plain, optim_plain, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor, Nx, Ny, Nz, epochs=config["epochs"], gamma=config["gamma"])
    rho_plain_gcc, gz_pred_plain_mps2 = evaluate_model(model_plain, coords_tensor, G_tensor, Nx, Ny, Nz)
    model_pos = DensityModel(use_posenc=True, num_freqs=4).to(device)
    optim_pos = torch.optim.Adam(model_pos.parameters(), lr=config["lr"])
    hist_pos = train(model_pos, optim_pos, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor, Nx, Ny, Nz, epochs=config["epochs"], gamma=config["gamma"])
    rho_pos_gcc, gz_pred_pos_mps2 = evaluate_model(model_pos, coords_tensor, G_tensor, Nx, Ny, Nz)
    os.makedirs("plots", exist_ok=True)
    plot_two_row_summary(
        rho_true_flat_gcc=rho_true_flat_gcc.cpu(),
        rho_plain_gcc=rho_plain_gcc,
        rho_pos_gcc=rho_pos_gcc,
        gz_obs_mps2=gz_obs_mps2.cpu().numpy(),
        gz_pred_plain_mps2=gz_pred_plain_mps2.numpy(),
        gz_pred_pos_mps2=gz_pred_pos_mps2.numpy(),
        hist_plain=hist_plain,
        hist_pos=hist_pos,
        grid_coords=grid_coords,
        obs_points=obs_points_np,
        Nx=Nx, Ny=Ny, Nz=Nz,
        save_path="plots/PureMLPvsPosEncMLP.png",
        z_index=None
    )

if __name__ == "__main__":
    main()
