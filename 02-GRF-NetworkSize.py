import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
from matplotlib import patheffects as pe

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
    def __init__(self, num_freqs=4, include_input=True):
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

def build_mlp(input_dim, hidden_sizes, output_dim=1):
    layers = []
    last = input_dim
    for h in hidden_sizes:
        layers += [nn.Linear(last, h), nn.LeakyReLU(0.01)]
        last = h
    layers += [nn.Linear(last, output_dim), nn.Sigmoid()]
    return nn.Sequential(*layers)

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class DensityModel(nn.Module):
    def __init__(self, hidden_sizes, num_freqs=4, min_density=1.6, max_density=3.5):
        super().__init__()
        self.positional_encoding = PositionalEncoding(num_freqs=num_freqs)
        input_dim = 3 * (1 + 2 * num_freqs)
        self.density_net = build_mlp(input_dim, hidden_sizes, 1)
        self.min_density = float(min_density)
        self.max_density = float(max_density)
    def forward(self, coords):
        feats = self.positional_encoding(coords)
        norm_density = self.density_net(feats)
        return self.min_density + norm_density * (self.max_density - self.min_density)

def train(model, optimizer, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor, Nx, Ny, Nz, epochs, gamma=1.0):
    model.train()
    history = []
    t0 = time.time()
    for _ in range(epochs):
        optimizer.zero_grad()
        rho_pred_gcc = model(coords_tensor).view(Nx * Ny * Nz)
        rho_pred_kgm3 = rho_pred_gcc * RHO_GC_TO_KGM3
        gz_pred_mps2 = G_tensor @ rho_pred_kgm3.unsqueeze(1)
        res_w = (gz_pred_mps2 - gz_obs_mps2) / sigma_mps2
        loss = gamma * F.mse_loss(res_w, torch.zeros_like(res_w))
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    t1 = time.time() - t0
    return {"gravity": history, "train_time_s": t1}

@torch.no_grad()
def evaluate_model(model, coords_tensor, G_tensor, Nx, Ny, Nz):
    model.eval()
    rho_pred_gcc = model(coords_tensor).flatten()
    gz_pred_mps2 = G_tensor @ (rho_pred_gcc * RHO_GC_TO_KGM3).unsqueeze(1)
    return rho_pred_gcc.cpu(), gz_pred_mps2.cpu()

def rmse(a, b):
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))

def plot_size_comparison(figpath, grid_coords, rho_true_flat_gcc, models_results, Nx, Ny, Nz, z_index=None):
    z_index = Nz // 2 if z_index is None else z_index
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    xs = sorted([(res["n_params"], label) for label, res in models_results.items()], key=lambda t: t[0])
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    label_to_color = {lab: colors[i % len(colors)] for i, (_, lab) in enumerate(xs)}
    ax = axes[0, 0]
    for _, label in xs:
        ax.plot(models_results[label]["history"]["gravity"], label=label, linewidth=2.5)
    ax.set_title("Training Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Whitened MSE")
    ax.set_yscale('log')
    ax.legend(title="Model", fontsize=9)
    ax = axes[0, 1]
    for nparams, label in xs:
        s = ax.scatter(nparams, models_results[label]["density_rmse_gcc"], s=160, edgecolors='k', linewidths=1.2, c=[label_to_color[label]])
        txt = ax.annotate(label, (nparams, models_results[label]["density_rmse_gcc"]), textcoords="offset points", xytext=(6,6), fontsize=10, color='k')
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xscale('log')
    ax.set_title("Density RMSE")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("RMSE (g/cm³)")
    ax = axes[0, 2]
    for nparams, label in xs:
        s = ax.scatter(nparams, models_results[label]["data_rmse_mgal"], s=160, edgecolors='k', linewidths=1.2, c=[label_to_color[label]])
        txt = ax.annotate(label, (nparams, models_results[label]["data_rmse_mgal"]), textcoords="offset points", xytext=(6,6), fontsize=10, color='k')
        txt.set_path_effects([pe.withStroke(linewidth=3, foreground="white")])
    ax.set_xscale('log')
    ax.set_title("Data Misfit RMSE")
    ax.set_xlabel("Parameters")
    ax.set_ylabel("RMSE (mGal)")
    x_coords = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y_coords = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    rho_true_slice = rho_true_flat_gcc.view(Nx, Ny, Nz)[:, :, z_index].cpu().numpy()
    im0 = axes[1, 0].imshow(rho_true_slice.T, origin='lower', extent=extent, cmap='viridis', vmin=1.6, vmax=3.5, aspect='equal')
    axes[1, 0].set_title(f"True density (z={z_index})")
    axes[1, 0].set_xlabel("x (m)")
    axes[1, 0].set_ylabel("y (m)")
    fig.colorbar(im0, ax=axes[1, 0], label="g/cm³")
    smallest = xs[0][1]
    largest = xs[-1][1]
    rho_small = models_results[smallest]["rho_pred_gcc"].view(Nx, Ny, Nz)[:, :, z_index].numpy()
    im1 = axes[1, 1].imshow(rho_small.T, origin='lower', extent=extent, cmap='viridis', vmin=1.6, vmax=3.5, aspect='equal')
    axes[1, 1].set_title(f"Inverted density ({smallest})")
    axes[1, 1].set_xlabel("x (m)")
    axes[1, 1].set_ylabel("y (m)")
    fig.colorbar(im1, ax=axes[1, 1], label="g/cm³")
    rho_large = models_results[largest]["rho_pred_gcc"].view(Nx, Ny, Nz)[:, :, z_index].numpy()
    im2 = axes[1, 2].imshow(rho_large.T, origin='lower', extent=extent, cmap='viridis', vmin=1.6, vmax=3.5, aspect='equal')
    axes[1, 2].set_title(f"Inverted density ({largest})")
    axes[1, 2].set_xlabel("x (m)")
    axes[1, 2].set_ylabel("y (m)")
    fig.colorbar(im2, ax=axes[1, 2], label="g/cm³")
    os.makedirs(os.path.dirname(figpath), exist_ok=True)
    plt.savefig(figpath, dpi=300, bbox_inches="tight", pad_inches=0.05)
    plt.close()

def main():
    set_seed(41)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
        sigma_mps2 = 0.01 * gz_true_mps2.std()
        noise = sigma_mps2 * torch.randn_like(gz_true_mps2)
        gz_obs_mps2 = gz_true_mps2 + noise
    archs = {
        "XS": [128, 16],
        "S":  [128, 64],
        "M":  [128, 128, 128],
        "L":  [256, 256, 256],
    }
    results = {}
    for label, hidden in archs.items():
        model = DensityModel(hidden_sizes=hidden, num_freqs=4).to(device)
        n_params = count_params(model)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        hist = train(model, optimizer, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor, Nx, Ny, Nz, epochs=500, gamma=1.0)
        rho_pred_gcc, gz_pred_mps2 = evaluate_model(model, coords_tensor, G_tensor, Nx, Ny, Nz)
        dens_rmse = rmse(rho_pred_gcc.numpy(), rho_true_flat_gcc.cpu().numpy())
        data_rmse_mgal = rmse(gz_pred_mps2.numpy().ravel()*MGAL_PER_MPS2, gz_obs_mps2.cpu().numpy().ravel()*MGAL_PER_MPS2)
        results[label] = {
            "n_params": n_params,
            "history": hist,
            "rho_pred_gcc": rho_pred_gcc,
            "density_rmse_gcc": dens_rmse,
            "data_rmse_mgal": data_rmse_mgal
        }
    os.makedirs("plots", exist_ok=True)
    plot_size_comparison("plots/NetworkSizeComparison.png", grid_coords, rho_true_flat_gcc, results, Nx, Ny, Nz, z_index=None)

if __name__ == "__main__":
    main()
