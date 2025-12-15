#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
05-TestingNoiseSensitivity.py
--------------------------------
Demonstrate the effect of **noise type** and **noise level** on GRF-based gravity inversion
with an implicit neural representation (INR). The script reuses the same grid and forward
physics used elsewhere in this repo (rectangular-prism kernel, dense G).

What it does
------------
1) Builds a GRF density model on a 40×40×20 grid (500 m spacing).
2) Assembles dense sensitivity matrix G for gz at a 40×40 surface grid.
3) For each noise type and noise level:
   - Corrupts gz_true with the specified noise (σ = level × std(gz_true)).
   - Trains an INR (PosEnc MLP) by minimizing whitened MSE.
   - Records metrics: density RMSE (g/cc), data RMSE (mGal), and residual spatial
     neighbor-autocorrelation (lag-1, 4-neighbor).
4) Saves simple line charts vs. noise level per noise type.

Noise types included
--------------------
- 'gaussian_iid' : N(0, σ²) independent samples.
- 'laplace'      : Laplace(0, b) with b chosen so std = σ (heavy-tailed).
- 'correlated'   : stationary Gaussian field with correlation length Lc (via FFT shaping).
- 'outliers'     : iid Gaussian noise plus sparse gross errors at a given fraction.

Usage
-----
$ python 05-TestingNoiseSensitivity.py
(Adjust CONFIG at the top as needed.)

Outputs
-------
- plots/Noise_Sensitivity_Combined.png

Notes
-----
- Default uses a single seed for speed. For robust statistics, set CONFIG['seeds'] to a list
  of multiple integers (e.g., [41, 42, 43, 44, 45]).
- The correlated-noise case intentionally trains with a diagonal Wd (σ⁻¹) to demonstrate
  residual spatial structure when noise is non-diagonal.
"""
import os
import time
import math
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------
# Constants
# ---------------------------
MGAL_PER_MPS2 = 1e5
RHO_GC_TO_KGM3 = 1000.0

# ---------------------------
# Repro / seeds
# ---------------------------
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

# ---------------------------
# Gravity kernel (gz)
# ---------------------------
def A_integral_torch(x, y, z):
    """
    Standard closed-form vertical component integral parts for a rectangular prism.
    Returns quantity whose linear combo gives potential; we'll multiply by Gamma outside.
    """
    Gamma = 6.67430e-11  # m^3 kg^-1 s^-2
    eps = 1e-12
    r = torch.sqrt(x**2 + y**2 + z**2)
    r_safe, z_safe = torch.clamp(r, min=eps), torch.clamp(z, min=eps)
    log_y_r, log_x_r = torch.log(torch.abs(y + r_safe)), torch.log(torch.abs(x + r_safe))
    arctan_term = torch.arctan((x * y) / (z_safe * r_safe))
    return -Gamma * (x * log_y_r + y * log_x_r - z * arctan_term)

@torch.inference_mode()
def construct_sensitivity_matrix_G_torch(cell_grid, data_points, d1, d2, device):
    """
    Dense sensitivity matrix G of shape (N_data, N_cells) for gz.
    cell_grid: (Ncells, 4) with [cx, cy, cz, dz_half]
    data_points: (Nobs, 3) with [ox, oy, oz]
    """
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
    A = (A_integral_torch(x2, y2, z2) - A_integral_torch(x2, y2, z1)
         - A_integral_torch(x2, y1, z2) + A_integral_torch(x2, y1, z1)
         - A_integral_torch(x1, y2, z2) + A_integral_torch(x1, y2, z1)
         + A_integral_torch(x1, y1, z2) - A_integral_torch(x1, y1, z1))
    return A  # already includes Gamma from A_integral_torch

# ---------------------------
# GRF generator (true density)
# ---------------------------
def generate_grf_torch(nx, ny, nz, dx, dy, dz, lambda_val, nu, sigma, device):
    kx = torch.fft.fftfreq(nx, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(ny, d=dy, device=device) * 2 * torch.pi
    kz = torch.fft.fftfreq(nz, d=dz, device=device) * 2 * torch.pi
    Kx, Ky, Kz = torch.meshgrid(kx, ky, kz, indexing='ij')
    k_squared = Kx**2 + Ky**2 + Kz**2
    # Matérn-like power spectrum
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

# ---------------------------
# Positional encoding INR
# ---------------------------
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

# ---------------------------
# Training & evaluation
# ---------------------------
def train(model, optimizer, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor,
          Nx, Ny, Nz, epochs, gamma=1.0):
    model.train()
    history = []
    for _ in range(epochs):
        optimizer.zero_grad()
        rho_pred_gcc = model(coords_tensor).view(Nx * Ny * Nz)
        rho_pred_kgm3 = rho_pred_gcc * RHO_GC_TO_KGM3
        gz_pred_mps2 = G_tensor @ rho_pred_kgm3.unsqueeze(1)
        res_w = (gz_pred_mps2 - gz_obs_mps2) / sigma_mps2
        loss = gamma * F.mse_loss(res_w, torch.zeros_like(res_w))
        loss.backward()
        optimizer.step()
        history.append(float(loss.item()))
    return history

@torch.no_grad()
def evaluate_model(model, coords_tensor, G_tensor, Nx, Ny, Nz):
    model.eval()
    rho_pred_gcc = model(coords_tensor).flatten()
    gz_pred_mps2 = G_tensor @ (rho_pred_gcc * RHO_GC_TO_KGM3).unsqueeze(1)
    return rho_pred_gcc.cpu(), gz_pred_mps2.cpu()

def rmse(a, b) -> float:
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    return float(np.sqrt(np.mean((a - b) ** 2)))

# ---------------------------
# Noise models
# ---------------------------
def add_noise_gaussian_iid(sig, std_scalar):
    return sig + std_scalar * torch.randn_like(sig)

def add_noise_laplace(sig, std_scalar):
    # Laplace(0, b) has std = sqrt(2) * b
    b = std_scalar / math.sqrt(2.0)
    u = torch.rand_like(sig) - 0.5  # in (-0.5, 0.5)
    lap = -b * torch.sign(u) * torch.log1p(-2.0 * torch.abs(u))
    return sig + lap

def add_noise_correlated(sig, std_scalar, Nx_obs, Ny_obs, dx, dy, Lc=2500.0, nu=1.0, device='cpu'):
    """
    Construct 2D Gaussian correlated noise via spectral shaping.
    Power spectrum ~ (k^2 + k0^2)^(-nu - 1), with k0 ~ 1/Lc.
    """
    # reshape
    field = torch.randn(Nx_obs, Ny_obs, dtype=torch.complex64, device=device)
    kx = torch.fft.fftfreq(Nx_obs, d=dx, device=device) * 2 * torch.pi
    ky = torch.fft.fftfreq(Ny_obs, d=dy, device=device) * 2 * torch.pi
    Kx, Ky = torch.meshgrid(kx, ky, indexing='ij')
    k2 = Kx**2 + Ky**2
    k0 = (1.0 / max(Lc, 1.0))
    P = (k2 + k0**2)**(-nu - 1.0)  # 2D analogue
    P[0, 0] = 0.0
    shaped = field * torch.sqrt(P)
    real = torch.real(torch.fft.ifftn(shaped))
    real = (real - real.mean()) / (real.std() + 1e-9)
    real = std_scalar * real
    return sig + real.reshape(-1, 1)

def add_noise_outliers(sig, std_scalar, frac=0.02, scale=6.0):
    """
    iid Gaussian noise with additional sparse gross errors at 'frac' fraction of stations.
    Outliers ~ N(0, (scale*std_scalar)^2).
    """
    noisy = sig + std_scalar * torch.randn_like(sig)
    N = noisy.numel()
    k = max(1, int(round(frac * N)))
    idx = torch.randperm(N)[:k]
    outliers = scale * std_scalar * torch.randn(k, device=noisy.device).unsqueeze(1)
    noisy[idx] += outliers
    return noisy

# ---------------------------
# Residual neighbor autocorrelation metric (lag-1)
# ---------------------------
def residual_neighbor_autocorr(residuals_vec, Nx_obs, Ny_obs) -> float:
    """
    Compute correlation between residuals and the mean of 4-neighbors on a 2D grid.
    Edges use available neighbors (average of existing ones).
    Returns Pearson correlation coefficient.
    """
    R = residuals_vec.reshape(Nx_obs, Ny_obs)
    # 4-neighbor averages
    up    = torch.roll(R, shifts=1,  dims=0)
    down  = torch.roll(R, shifts=-1, dims=0)
    left  = torch.roll(R, shifts=1,  dims=1)
    right = torch.roll(R, shifts=-1, dims=1)
    # For edges, rolled values include wrap-around; fix by replacing wrap edges with original
    up[0, :]    = R[0, :]
    down[-1, :] = R[-1, :]
    left[:, 0]  = R[:, 0]
    right[:, -1]= R[:, -1]
    neigh_mean = 0.25 * (up + down + left + right)
    r = R.flatten()
    n = neigh_mean.flatten()
    r = r - r.mean()
    n = n - n.mean()
    denom = torch.sqrt((r**2).sum() * (n**2).sum()) + 1e-12
    return float((r @ n) / denom)

# ---------------------------
# Plot helpers
# ---------------------------
def plot_all_metrics_combined(noise_levels, dens_series, data_series, ac_series, save_path):
    """Create a single figure with all three metrics as subplots"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Define colors for each noise type for consistency
    colors = {'gaussian_iid': 'blue', 'laplace': 'red', 'correlated': 'green', 'outliers': 'orange'}
    
    # Plot 1: Density RMSE
    ax1 = axes[0]
    for noise_type, values in dens_series.items():
        ax1.plot(noise_levels, values, marker='o', linewidth=1, 
                label=noise_type, color=colors.get(noise_type, 'black'))
    ax1.set_xlabel('Noise level (fraction of std(gz_true))')
    ax1.set_ylabel('Density RMSE (g/cm³)')
    ax1.set_title('Density RMSE vs Noise Level')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot 2: Data RMSE
    ax2 = axes[1]
    for noise_type, values in data_series.items():
        ax2.plot(noise_levels, values, marker='s', linewidth=1,
                label=noise_type, color=colors.get(noise_type, 'black'))
    ax2.set_xlabel('Noise level (fraction of std(gz_true))')
    ax2.set_ylabel('Data RMSE (mGal)')
    ax2.set_title('Data RMSE vs Noise Level')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Autocorrelation
    ax3 = axes[2]
    for noise_type, values in ac_series.items():
        ax3.plot(noise_levels, values, marker='^', linewidth=1,
                label=noise_type, color=colors.get(noise_type, 'black'))
    ax3.set_xlabel('Noise level (fraction of std(gz_true))')
    ax3.set_ylabel('Residual neighbor autocorr')
    ax3.set_title('Residual Spatial Structure vs Noise Level')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# ---------------------------
# Main experiment
# ---------------------------
def main():
    CONFIG = {
        "seed_base": 41,
        "seeds": [41],          # for speed; change to multiple seeds for robust stats
        "epochs": 300,
        "lr": 1e-3,
        "gamma": 1.0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "Nx": 40, "Ny": 40, "Nz": 20,
        "dx": 500.0, "dy": 500.0, "dz": 500.0,
        "grf_lambda": 5000.0, "grf_nu": 1.5, "grf_sigma": 2.0,
        "noise_levels": [0.005, 0.01, 0.02, 0.05],  # relative to std(gz_true)
        "noise_types": ["gaussian_iid", "laplace", "correlated", "outliers"],
        # correlated noise settings
        "corr_Lc": 2500.0, "corr_nu": 1.0,
        # outliers settings
        "outlier_frac": 0.02, "outlier_scale": 6.0,
    }

    device = torch.device(CONFIG["device"])
    set_seed(CONFIG["seed_base"])

    # --- Model grid ---
    Nx, Ny, Nz = CONFIG["Nx"], CONFIG["Ny"], CONFIG["Nz"]
    dx, dy, dz = CONFIG["dx"], CONFIG["dy"], CONFIG["dz"]
    x = np.linspace(0, (Nx - 1) * dx, Nx)
    y = np.linspace(0, (Ny - 1) * dy, Ny)
    z = np.linspace(0, (Nz - 1) * dz, Nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    grid_coords = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)

    coords_mean = grid_coords.mean(axis=0, keepdims=True)
    coords_std = grid_coords.std(axis=0, keepdims=True)
    coords_norm = (grid_coords - coords_mean) / coords_std
    coords_tensor = torch.tensor(coords_norm, dtype=torch.float32, device=device, requires_grad=True)

    dz_half = dz / 2.0
    cell_grid_np = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid_tensor = torch.tensor(cell_grid_np, dtype=torch.float32, device=device)

    # --- Observation grid on surface (z=0) ---
    X_obs, Y_obs = np.meshgrid(x, y, indexing='ij')
    obs_points_np = np.column_stack([X_obs.ravel(), Y_obs.ravel(), np.zeros_like(X_obs.ravel())])
    obs_points_tensor = torch.tensor(obs_points_np, dtype=torch.float32, device=device)

    # --- True GRF model in g/cc, limited to [1.6, 3.5] ---
    rho_true_3d_gcc = generate_grf_torch(Nx, Ny, Nz, dx, dy, dz,
                                         CONFIG["grf_lambda"], CONFIG["grf_nu"], CONFIG["grf_sigma"], device)
    # rescale to [1.6, 3.5] g/cc for consistency with other scripts
    min_val, max_val = torch.min(rho_true_3d_gcc), torch.max(rho_true_3d_gcc)
    rho_true_3d_gcc = 1.6 + (rho_true_3d_gcc - min_val) * ((3.5 - 1.6) / (max_val - min_val))
    rho_true_flat_gcc = rho_true_3d_gcc.flatten()

    # --- Sensitivity matrix ---
    print("Assembling sensitivity matrix G ...")
    t0 = time.time()
    G_tensor = construct_sensitivity_matrix_G_torch(cell_grid_tensor, obs_points_tensor, dx, dy, device)
    G_tensor = G_tensor.clone().detach().requires_grad_(False)
    print(f"G shape = {tuple(G_tensor.shape)}, build time = {time.time() - t0:.2f}s")

    # --- True data ---
    with torch.no_grad():
        gz_true_mps2 = G_tensor @ (rho_true_flat_gcc * RHO_GC_TO_KGM3).unsqueeze(1)

    # --- Network ---
    hidden = [128, 128, 128]   # "M" capacity from previous experiments
    model_template = lambda: DensityModel(hidden_sizes=hidden, num_freqs=4).to(device)

    # --- Book-keeping ---
    results = []  # list of dict rows
    noise_levels = CONFIG["noise_levels"]
    noise_types  = CONFIG["noise_types"]

    for ntype in noise_types:
        dens_rmse_levels: List[float] = []
        data_rmse_levels: List[float] = []
        ac_levels: List[float] = []

        for nl in noise_levels:
            # compute sigma from true data std
            sigma_mps2 = nl * gz_true_mps2.std()
            Wd = 1.0 / sigma_mps2  # diagonal std (scalar)

            # create noisy data according to type
            if ntype == "gaussian_iid":
                gz_obs_mps2 = add_noise_gaussian_iid(gz_true_mps2, sigma_mps2)

            elif ntype == "laplace":
                gz_obs_mps2 = add_noise_laplace(gz_true_mps2, sigma_mps2)

            elif ntype == "correlated":
                gz_obs_mps2 = add_noise_correlated(
                    gz_true_mps2, sigma_mps2, Nx, Ny, dx, dy,
                    Lc=CONFIG["corr_Lc"], nu=CONFIG["corr_nu"], device=device)

            elif ntype == "outliers":
                gz_obs_mps2 = add_noise_outliers(
                    gz_true_mps2, sigma_mps2, frac=CONFIG["outlier_frac"], scale=CONFIG["outlier_scale"])

            else:
                raise ValueError(f"Unknown noise type: {ntype}")

            # Single-seed training (or loop for multiple seeds if desired)
            seed_metrics = []
            for seed in CONFIG["seeds"]:
                set_seed(seed)
                model = model_template()
                optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG["lr"])
                history = train(model, optimizer, coords_tensor, gz_obs_mps2, sigma_mps2, G_tensor,
                                Nx, Ny, Nz, epochs=CONFIG["epochs"], gamma=CONFIG["gamma"])
                rho_pred_gcc, gz_pred_mps2 = evaluate_model(model, coords_tensor, G_tensor, Nx, Ny, Nz)

                # Metrics
                dens_rmse = rmse(rho_pred_gcc.numpy(), rho_true_flat_gcc.cpu().numpy())  # g/cc
                data_rmse_mgal = rmse(
                    gz_pred_mps2.numpy().ravel() * MGAL_PER_MPS2,
                    gz_obs_mps2.detach().cpu().numpy().ravel() * MGAL_PER_MPS2
                )
                residuals = (gz_pred_mps2 - gz_obs_mps2).detach()
                ac = residual_neighbor_autocorr(residuals.squeeze(1), Nx, Ny)

                seed_metrics.append((dens_rmse, data_rmse_mgal, ac))

            # aggregate (mean across seeds)
            dens_rmse_levels.append(float(np.mean([m[0] for m in seed_metrics])))
            data_rmse_levels.append(float(np.mean([m[1] for m in seed_metrics])))
            ac_levels.append(float(np.mean([m[2] for m in seed_metrics])))

            # store row(s)
            for (dens_rmse, data_rmse_mgal, ac), seed in zip(seed_metrics, CONFIG["seeds"]):
                results.append({
                    "noise_type": ntype,
                    "noise_level": nl,
                    "seed": seed,
                    "density_rmse_gcc": dens_rmse,
                    "data_rmse_mgal": data_rmse_mgal,
                    "residual_neighbor_ac": ac
                })

        # After finishing a noise type, plot its lines across levels
        x = noise_levels
        # We'll accumulate across types later; here we just continue

    # --- Aggregate for plotting ---
    # Compute mean over seeds for each (noise_type, noise_level)
    from collections import defaultdict
    agg_dens = defaultdict(list)
    agg_data = defaultdict(list)
    agg_ac   = defaultdict(list)
    for row in results:
        key = (row["noise_type"], row["noise_level"])
        agg_dens[key].append(row["density_rmse_gcc"])
        agg_data[key].append(row["data_rmse_mgal"])
        agg_ac[key].append(row["residual_neighbor_ac"])

    noise_levels_sorted = sorted({nl for _, nl in agg_dens.keys()})
    dens_series = {}
    data_series = {}
    ac_series   = {}
    for ntype in CONFIG["noise_types"]:
        dens_series[ntype] = [float(np.mean(agg_dens[(ntype, nl)])) for nl in noise_levels_sorted]
        data_series[ntype] = [float(np.mean(agg_data[(ntype, nl)])) for nl in noise_levels_sorted]
        ac_series[ntype]   = [float(np.mean(agg_ac[(ntype, nl)]))   for nl in noise_levels_sorted]

    # --- Plots ---
    os.makedirs("plots", exist_ok=True)
    plot_all_metrics_combined(noise_levels_sorted, dens_series, data_series, ac_series,
                             "plots/Noise_Sensitivity_Combined.png")
    
    print("Saved combined plot to 'plots/Noise_Sensitivity_Combined.png'")

if __name__ == "__main__":
    main()
