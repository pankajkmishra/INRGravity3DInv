# Author: Pankaj K Mishra
# Date: July 31, 2025

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import random
import imageio.v2 as imageio

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
    print(f"Random seed set to {seed}")

def A_integral_torch(x, y, z):
    Gamma = 6.67430e-11
    eps = 1e-12
    r = torch.sqrt(x**2 + y**2 + z**2)
    r_safe, z_safe = torch.clamp(r, min=eps), torch.clamp(z, min=eps)
    log_y_r, log_x_r = torch.log(torch.abs(y + r_safe)), torch.log(torch.abs(x + r_safe))
    arctan_term = torch.arctan((x * y) / (z_safe * r_safe))
    return -Gamma * (x * log_y_r + y * log_x_r - z * arctan_term)

def construct_sensitivity_matrix_G_torch(cell_grid, data_points, d1, d2, device):
    cell_x, cell_y, cell_z, cell_dz_half = cell_grid[:, 0].unsqueeze(0), cell_grid[:, 1].unsqueeze(0), cell_grid[:, 2].unsqueeze(0), cell_grid[:, 3].unsqueeze(0)
    obs_x, obs_y, obs_z = data_points[:, 0].unsqueeze(1), data_points[:, 1].unsqueeze(1), data_points[:, 2].unsqueeze(1)

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

def main():
    config = {
        "gamma": 1.0,
        "epochs": 500,
        "lr": 0.001,
        "noise_level": 0.01,
        "seed": 41,
        "model_path": "model/density_inverter_model.pt",
        "gif_interval": 1, # Save a frame every 10 epochs
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config["seed"])

    print("--- Configuration ---")
    for key, value in config.items():
        print(f"{key:<22}: {value}")
    print(f"{'device':<22}: {device}")
    print("---------------------")
    
    # Create directories for plots and GIF frames
    plots_dir = "plots"
    gif_frame_dir = os.path.join(plots_dir, "gif_frames")
    os.makedirs(gif_frame_dir, exist_ok=True)


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

    print("Generating a true density model...")
    rho_true_3d = generate_grf_torch(Nx, Ny, Nz, dx, dy, dz, 5000.0, 1.5, 2.0, device)
    min_val, max_val = torch.min(rho_true_3d), torch.max(rho_true_3d)
    rho_true_3d = 1.6 + (rho_true_3d - min_val) * ((3.5 - 1.6) / (max_val - min_val))
    rho_true_flat = rho_true_3d.flatten()

    print("Pre-computing sensitivity matrix G...")
    start_time = time.time()
    G_tensor = construct_sensitivity_matrix_G_torch(cell_grid_tensor, obs_points_tensor, dx, dy, device)
    print(f"G matrix computation complete. Time: {time.time() - start_time:.2f}s.")

    print("Calculating observed data and adding noise...")
    with torch.no_grad():
        gz_true_clean = G_tensor @ rho_true_flat.unsqueeze(1)
        noise = torch.randn_like(gz_true_clean) * config["noise_level"] * torch.std(gz_true_clean)
        gz_obs_noisy = gz_true_clean + noise

    gz_mean, gz_std = gz_obs_noisy.mean(), gz_obs_noisy.std()
    gz_obs_norm_target = (gz_obs_noisy - gz_mean) / gz_std

    model = DensityModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])

    print(f"Training model with gamma={config['gamma']}...")
    loss_history = train(
        model, optimizer, coords_tensor, gz_obs_norm_target,
        G_tensor, gz_mean, gz_std,
        Nx, Ny, Nz, config, rho_true_3d, grid_coords, gif_frame_dir
    )

    print("Evaluating final model...")
    rho_inverted, gz_pred = evaluate_model(
        model, coords_tensor, G_tensor,
    )

    print("Generating plots...")
    if loss_history:
        plot_loss_curves(loss_history, os.path.join(plots_dir, "training_losses.png"))

    plot_final_results(
        rho_inverted, rho_true_3d, gz_pred.cpu(),
        gz_obs_noisy.cpu(), grid_coords,
        Nx, Ny, Nz, os.path.join(plots_dir, "final_results.png")
    )

    plot_data_scatter(
        gz_obs_noisy.cpu().numpy().flatten(),
        gz_pred.cpu().numpy().flatten(),
        obs_points_np, os.path.join(plots_dir, "data_scatter.png")
    )
    
    print("Creating inversion evolution GIF...")
    gif_path = os.path.join(plots_dir, "inversion_evolution.gif")
    create_inversion_gif(gif_frame_dir, gif_path, cleanup=True)
    print(f"GIF saved to {gif_path}")

    print("Done.")

def create_network(input_dim, output_dim=1):
    return nn.Sequential(
        nn.Linear(input_dim, 256), nn.LeakyReLU(0.01),
        nn.Linear(256, 128), nn.LeakyReLU(0.01),
        nn.Linear(128, 64), nn.LeakyReLU(0.01),
        nn.Linear(64, output_dim),
        nn.Sigmoid()
    )

class DensityModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.density_net = create_network(3, 1)
        self.min_density = 1.6
        self.max_density = 3.5

    def forward(self, coords):
        norm_density = self.density_net(coords)
        return self.min_density + norm_density * (self.max_density - self.min_density)

def train(model, optimizer, coords_tensor, gz_obs_norm_target,
          G_tensor, gz_mean, gz_std,
          Nx, Ny, Nz, config, rho_true_3d, grid_coords, frame_dir):
    model.train()
    history = {"total": [], "gravity": []}

    gamma = config["gamma"]
    epochs = config["epochs"]
    gif_interval = config["gif_interval"]
    
    # Prepare details for plotting frames
    z_index = Nz // 2
    rho_true_slice = rho_true_3d.view(Nx, Ny, Nz)[:, :, z_index].detach().cpu().numpy()
    x_coords = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y_coords = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    
    for epoch in range(epochs):
        optimizer.zero_grad()

        rho_pred_physical_flat = model(coords_tensor).view(Nx*Ny*Nz)
        gz_pred = G_tensor @ rho_pred_physical_flat.unsqueeze(1)
        gz_pred_norm = (gz_pred - gz_mean) / gz_std

        gravity_loss = gamma * F.mse_loss(gz_pred_norm, gz_obs_norm_target)
        total_loss = gravity_loss
        total_loss.backward()
        optimizer.step()

        history["total"].append(total_loss.item())
        history["gravity"].append(gravity_loss.item())

        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch}/{epochs}, Avg Gravity Misfit Loss: {history['gravity'][-1]:.6f}")
            
        # --- GIF Frame Generation ---
        if epoch % gif_interval == 0 or epoch == epochs - 1:
            with torch.no_grad():
                rho_pred_physical = model(coords_tensor).detach()
                rho_pred_slice = rho_pred_physical.view(Nx, Ny, Nz)[:, :, z_index].cpu().numpy()
                frame_path = os.path.join(frame_dir, f"frame_{epoch:04d}.png")
                plot_inversion_frame(rho_true_slice, rho_pred_slice, epoch, extent, frame_path)
                
    return history

def evaluate_model(model, coords_tensor, G_tensor):
    model.eval()
    with torch.no_grad():
        rho_pred_physical = model(coords_tensor).flatten()
        gz_pred = G_tensor @ rho_pred_physical.unsqueeze(1)
        rho_pred = rho_pred_physical.cpu()
        gz_pred = gz_pred.cpu()
    return rho_pred, gz_pred

def plot_inversion_frame(rho_true_slice, rho_pred_slice, epoch, extent, save_path):
    """Plots a single frame for the GIF, comparing true and inverted density."""
    vmin_rho, vmax_rho = 1.6, 3.5
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # True Density
    im1 = axes[0].imshow(rho_true_slice.T, origin='lower', cmap='coolwarm', vmin=vmin_rho, vmax=vmax_rho, extent=extent, aspect='auto')
    axes[0].set_title('True Density')
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[0], shrink=0.8, label='Density (g/cm³)')

    # Inverted Density
    im2 = axes[1].imshow(rho_pred_slice.T, origin='lower', cmap='coolwarm', vmin=vmin_rho, vmax=vmax_rho, extent=extent, aspect='auto')
    axes[1].set_title(f'Inverted Density (Epoch {epoch})')
    axes[1].set_xlabel("x (m)")
    axes[1].set_yticklabels([])
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label='Density (g/cm³)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150) # Use a moderate DPI for GIFs
    plt.close(fig)

def create_inversion_gif(frame_dir, gif_path, cleanup=True):
    """Creates a GIF from saved frames and optionally cleans them up."""
    filenames = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith('.png')])
    if not filenames:
        print("No frames found to create a GIF.")
        return
        
    with imageio.get_writer(gif_path, mode='I', duration=0.2, loop=0) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    if cleanup:
        for filename in filenames:
            os.remove(filename)
        # Try to remove the directory if it's empty
        try:
            os.rmdir(frame_dir)
        except OSError:
            pass # Directory not empty, do nothing

def plot_loss_curves(history, save_path):
    plt.figure(figsize=(10, 6))
    plt.plot(history['gravity'], label='Gravity Misfit Loss', color='black')
    plt.xlabel("Epoch")
    plt.ylabel("Average Loss")
    plt.title("Training Loss")
    plt.yscale('log')
    plt.grid(True, which="both", ls="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_final_results(rho_pred, rho_true, gz_pred, gz_true, grid_coords, Nx, Ny, Nz, save_path, z_index=None):
    z_index = Nz // 2 if z_index is None else z_index

    rho_pred_slice = rho_pred.view(Nx, Ny, Nz)[:, :, z_index].detach().cpu().numpy()
    rho_true_slice = rho_true.view(Nx, Ny, Nz)[:, :, z_index].detach().cpu().numpy()
    vmin_rho, vmax_rho = 1.6, 3.5

    x_coords = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y_coords = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]

    gz_true_2d = gz_true.reshape(Nx, Ny).cpu().numpy()
    gz_pred_2d = gz_pred.reshape(Nx, Ny).cpu().numpy()
    misfit = gz_true_2d - gz_pred_2d

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    im1 = axes[0].imshow(rho_true_slice.T, origin='lower', cmap='coolwarm', vmin=vmin_rho, vmax=vmax_rho, extent=extent, aspect='auto')
    axes[0].set_title(f'True Density (z-slice: {z_index})')
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[0], shrink=0.8, label='Density (g/cm³)')

    im2 = axes[1].imshow(rho_pred_slice.T, origin='lower', cmap='coolwarm', vmin=vmin_rho, vmax=vmax_rho, extent=extent, aspect='auto')
    axes[1].set_title('Inverted Density')
    axes[1].set_xlabel("x (m)")
    axes[1].set_yticklabels([])
    fig.colorbar(im2, ax=axes[1], shrink=0.8, label='Density (g/cm³)')

    im3 = axes[2].imshow(misfit.T, origin='lower', cmap='coolwarm', extent=extent, aspect='auto')
    axes[2].set_title('Gravity Misfit (Observed - Predicted)')
    axes[2].set_xlabel("x (m)")
    axes[2].set_yticklabels([])
    fig.colorbar(im3, ax=axes[2], shrink=0.8, label='Gravity Residual (mGal)')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_data_scatter(gz_obs, gz_pred, obs_points, save_path):
    x_obs = obs_points[:, 0]
    y_obs = obs_points[:, 1]
    residuals = gz_obs - gz_pred

    vmin = min(gz_obs.min(), gz_pred.min())
    vmax = max(gz_obs.max(), gz_pred.max())
    res_max_abs = np.abs(residuals).max()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)

    sc1 = axes[0].scatter(x_obs, y_obs, c=gz_obs, cmap='coolwarm', vmin=vmin, vmax=vmax, s=10)
    axes[0].set_title('Observed Data')
    axes[0].set_xlabel('x (m)')
    axes[0].set_ylabel('y (m)')
    axes[0].set_aspect('equal', adjustable='box')
    fig.colorbar(sc1, ax=axes[0], label='Gravity (mGal)', shrink=0.8)

    sc2 = axes[1].scatter(x_obs, y_obs, c=gz_pred, cmap='coolwarm', vmin=vmin, vmax=vmax, s=10)
    axes[1].set_title('Predicted Data')
    axes[1].set_xlabel('x (m)')
    axes[1].set_aspect('equal', adjustable='box')
    axes[1].set_yticklabels([])
    fig.colorbar(sc2, ax=axes[1], label='Gravity (mGal)', shrink=0.8)

    sc3 = axes[2].scatter(x_obs, y_obs, c=residuals, cmap='coolwarm', vmin=-res_max_abs, vmax=res_max_abs, s=10)
    axes[2].set_title('Residuals (Observed - Predicted)')
    axes[2].set_xlabel('x (m)')
    axes[2].set_aspect('equal', adjustable='box')
    axes[2].set_yticklabels([])
    fig.colorbar(sc3, ax=axes[2], label='Gravity Residual (mGal)', shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

if __name__ == "__main__":

    main()
