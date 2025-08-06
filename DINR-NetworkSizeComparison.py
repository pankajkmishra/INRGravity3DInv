import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import time
import random
import seaborn as sns
from sklearn.metrics import r2_score
import pandas as pd

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

class PositionalEncoding(nn.Module):
    def __init__(self, num_freqs=10, include_input=True):
        super().__init__()
        self.include_input = include_input
        self.num_freqs = num_freqs
        self.freqs = 2.0 ** torch.arange(0, num_freqs)
        
    def forward(self, x):
        encoded = []
        if self.include_input:
            encoded.append(x)
            
        for freq in self.freqs:
            encoded.append(torch.sin(x * freq))
            encoded.append(torch.cos(x * freq))
            
        return torch.cat(encoded, dim=-1)

class DensityModel(nn.Module):
    def __init__(self, layer_sizes=[256, 128, 64], activation='leaky_relu'):
        super().__init__()
        self.positional_encoding = PositionalEncoding(num_freqs=10)
        input_dim = 3 * (1 + 2 * 10)
        
        layers = []
        prev_size = input_dim
        
        for size in layer_sizes:
            layers.append(nn.Linear(prev_size, size))
            if activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(0.01))
            elif activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            prev_size = size
        
        layers.append(nn.Linear(prev_size, 1))
        layers.append(nn.Sigmoid())
        
        self.density_net = nn.Sequential(*layers)
        self.min_density = 1.6
        self.max_density = 3.5

    def forward(self, coords):
        encoded_coords = self.positional_encoding(coords)
        norm_density = self.density_net(encoded_coords)
        return self.min_density + norm_density * (self.max_density - self.min_density)

def train_model(model, coords_tensor, gz_obs_norm_target, G_tensor, gz_mean, gz_std, Nx, Ny, Nz, config):
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    model.train()
    history = {"total": [], "gravity": []}
    
    gamma = config["gamma"]
    epochs = config["epochs"]

    for epoch in range(epochs):
        optimizer.zero_grad()

        rho_pred_physical = model(coords_tensor).view(Nx*Ny*Nz)
        gz_pred = G_tensor @ rho_pred_physical.unsqueeze(1)
        gz_pred_norm = (gz_pred - gz_mean) / gz_std

        gravity_loss = gamma * F.mse_loss(gz_pred_norm, gz_obs_norm_target)
        total_loss = gravity_loss

        total_loss.backward()
        optimizer.step()

        history["total"].append(total_loss.item())
        history["gravity"].append(gravity_loss.item())

        if epoch % 100 == 0:
            print(f"    Epoch {epoch}/{epochs}, Loss: {history['gravity'][-1]:.6f}")

    return history

def evaluate_model(model, coords_tensor, G_tensor, gz_mean, gz_std):
    model.eval()
    with torch.no_grad():
        rho_pred_physical = model(coords_tensor).flatten()
        gz_pred = G_tensor @ rho_pred_physical.unsqueeze(1)
        
        rho_pred = rho_pred_physical.cpu()
        gz_pred = gz_pred.cpu()

    return rho_pred, gz_pred

def calculate_metrics(rho_true, rho_pred, gz_true, gz_pred):
    rho_true_np = rho_true.cpu().numpy().flatten()
    rho_pred_np = rho_pred.cpu().numpy().flatten()
    gz_true_np = gz_true.cpu().numpy().flatten()
    gz_pred_np = gz_pred.cpu().numpy().flatten()
    
    density_rmse = np.sqrt(np.mean((rho_true_np - rho_pred_np)**2))
    gravity_rmse = np.sqrt(np.mean((gz_true_np - gz_pred_np)**2))
    density_r2 = r2_score(rho_true_np, rho_pred_np)
    gravity_r2 = r2_score(gz_true_np, gz_pred_np)
    
    return {
        'density_rmse': density_rmse,
        'gravity_rmse': gravity_rmse,
        'density_r2': density_r2,
        'gravity_r2': gravity_r2
    }

def run_network_size_experiment():
    config = {
        "gamma": 1.0,
        "epochs": 300,
        "lr": 0.001,
        "noise_level": 0.01,
        "seed": 42,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(config["seed"])

    print("Setting up experiment environment...")
    
    Nx, Ny, Nz = 30, 30, 15
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

    print("Generating true density model...")
    rho_true_3d = generate_grf_torch(Nx, Ny, Nz, dx, dy, dz, 5000.0, 1.5, 2.0, device)
    min_val, max_val = torch.min(rho_true_3d), torch.max(rho_true_3d)
    rho_true_3d = 1.6 + (rho_true_3d - min_val) * ((3.5 - 1.6) / (max_val - min_val))
    rho_true_flat = rho_true_3d.flatten()

    print("Computing sensitivity matrix...")
    G_tensor = construct_sensitivity_matrix_G_torch(cell_grid_tensor, obs_points_tensor, dx, dy, device)

    with torch.no_grad():
        gz_true_clean = G_tensor @ rho_true_flat.unsqueeze(1)
        noise = torch.randn_like(gz_true_clean) * config["noise_level"] * torch.std(gz_true_clean)
        gz_obs_noisy = gz_true_clean + noise

    gz_mean, gz_std = gz_obs_noisy.mean(), gz_obs_noisy.std()
    gz_obs_norm_target = (gz_obs_noisy - gz_mean) / gz_std

    network_configs = [
        {'name': 'Small', 'layers': [64, 32], 'color': '#1f77b4'},
        {'name': 'Medium-Small', 'layers': [128, 64], 'color': '#ff7f0e'},
        {'name': 'Medium', 'layers': [256, 128], 'color': '#2ca02c'},
        {'name': 'Medium-Large', 'layers': [256, 128, 64], 'color': '#d62728'},
        {'name': 'Large', 'layers': [512, 256, 128], 'color': '#9467bd'},
        {'name': 'Very Large', 'layers': [512, 256, 128, 64], 'color': '#8c564b'},
        {'name': 'Extra Large', 'layers': [1024, 512, 256], 'color': '#e377c2'}
    ]

    results = []
    all_histories = {}
    best_models = {}

    print("\nRunning experiments for different network sizes...")
    
    for i, net_config in enumerate(network_configs):
        print(f"\n--- Testing {net_config['name']} Network: {net_config['layers']} ---")
        
        model = DensityModel(layer_sizes=net_config['layers']).to(device)
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {num_params:,}")
        
        start_time = time.time()
        history = train_model(model, coords_tensor, gz_obs_norm_target, G_tensor, gz_mean, gz_std, Nx, Ny, Nz, config)
        training_time = time.time() - start_time
        
        rho_pred, gz_pred = evaluate_model(model, coords_tensor, G_tensor, gz_mean, gz_std)
        
        metrics = calculate_metrics(rho_true_flat, rho_pred, gz_obs_noisy, gz_pred)
        
        result = {
            'name': net_config['name'],
            'layers': str(net_config['layers']),
            'num_params': num_params,
            'training_time': training_time,
            'final_loss': history['gravity'][-1],
            'color': net_config['color'],
            **metrics
        }
        
        results.append(result)
        all_histories[net_config['name']] = history
        best_models[net_config['name']] = {
            'model': model,
            'rho_pred': rho_pred,
            'gz_pred': gz_pred
        }
        
        print(f"Training time: {training_time:.1f}s")
        print(f"Final loss: {result['final_loss']:.6f}")
        print(f"Density RMSE: {result['density_rmse']:.4f}")
        print(f"Gravity RMSE: {result['gravity_rmse']:.4f}")

    os.makedirs("publication_figures", exist_ok=True)
    
    create_comprehensive_plots(results, all_histories, best_models, rho_true_3d, gz_obs_noisy, 
                             grid_coords, Nx, Ny, Nz, network_configs)
    
    results_df = pd.DataFrame(results)
    
    print("\nExperiment completed! Results saved in 'publication_figures/' directory.")
    return results

def create_comprehensive_plots(results, all_histories, best_models, rho_true_3d, gz_obs_noisy, 
                             grid_coords, Nx, Ny, Nz, network_configs):
    
    fig = plt.figure(figsize=(20, 14))
    
    ax1 = plt.subplot(2, 4, 1)
    for i, result in enumerate(results):
        history = all_histories[result['name']]
        plt.plot(history['gravity'], label=result['name'], color=result['color'], linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Training Loss Curves', fontsize=14)
    plt.yscale('log')
    plt.legend(fontsize=9)
    
    ax2 = plt.subplot(2, 4, 2)
    num_params = [r['num_params'] for r in results]
    density_rmse = [r['density_rmse'] for r in results]
    colors = [r['color'] for r in results]
    plt.scatter(num_params, density_rmse, c=colors, s=100, alpha=0.7, edgecolors='black')
    for i, result in enumerate(results):
        plt.annotate(result['name'], (num_params[i], density_rmse[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Density RMSE (g/cm³)', fontsize=12)
    plt.title(' Density Accuracy', fontsize=14)
    plt.xscale('log')
    
    ax3 = plt.subplot(2, 4, 3)
    gravity_rmse = [r['gravity_rmse'] for r in results]
    plt.scatter(num_params, gravity_rmse, c=colors, s=100, alpha=0.7, edgecolors='black')
    for i, result in enumerate(results):
        plt.annotate(result['name'], (num_params[i], gravity_rmse[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Gravity RMSE (mGal)', fontsize=12)
    plt.title('Data Misfit', fontsize=14)
    plt.xscale('log')
    
    ax4 = plt.subplot(2, 4, 4)
    training_times = [r['training_time'] for r in results]
    plt.scatter(num_params, training_times, c=colors, s=100, alpha=0.7, edgecolors='black')
    for i, result in enumerate(results):
        plt.annotate(result['name'], (num_params[i], training_times[i]), 
                    xytext=(5, 5), textcoords='offset points', fontsize=9)
    plt.xlabel('Number of Parameters', fontsize=12)
    plt.ylabel('Training Time (seconds)', fontsize=12)
    plt.title('Training Time', fontsize=14)
    plt.xscale('log')
    
    z_index = Nz // 2
    rho_true_slice = rho_true_3d.view(Nx, Ny, Nz)[:, :, z_index].detach().cpu().numpy()
    x_coords = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
    y_coords = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
    extent = [x_coords.min(), x_coords.max(), y_coords.min(), y_coords.max()]
    
    ax5 = plt.subplot(2, 4, 5)
    im = plt.imshow(rho_true_slice.T, origin='lower', cmap='coolwarm', vmin=1.6, vmax=3.5, 
                    extent=extent, aspect='equal')
    plt.title('True density', fontsize=14)
    plt.xlabel('x (m)', fontsize=12)
    plt.ylabel('y (m)', fontsize=12)
    ax5.set_xlim(extent[0], extent[1])
    ax5.set_ylim(extent[2], extent[3])
    
    small_model = best_models['Small']
    medium_model = best_models['Medium']
    large_model = best_models['Large']
    
    models_to_show = [
        ('Small', small_model, 6),
        ('Medium', medium_model, 7),
        ('Large', large_model, 8)
    ]
    
    for name, model_data, subplot_idx in models_to_show:
        ax = plt.subplot(2, 4, subplot_idx)
        rho_pred_slice = model_data['rho_pred'].view(Nx, Ny, Nz)[:, :, z_index].detach().cpu().numpy()
        im = plt.imshow(rho_pred_slice.T, origin='lower', cmap='coolwarm', vmin=1.6, vmax=3.5, 
                        extent=extent, aspect='equal')
        plt.title(f'Inverted density ({name})', fontsize=14)
        plt.xlabel('x (m)', fontsize=12)
        if subplot_idx == 6:
            plt.ylabel('y (m)', fontsize=12)
        else:
            plt.ylabel('')
        ax.set_xlim(extent[0], extent[1])
        ax.set_ylim(extent[2], extent[3])
    
    plt.subplots_adjust(bottom=0.15)
    
    cbar_ax = fig.add_axes([0.1, 0.03, 0.8, 0.02])
    cbar = fig.colorbar(im, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Density (g/cm³)', fontsize=12)
    
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    plt.savefig("publication_figures/comprehensive_network_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()
    
 
    
    


if __name__ == "__main__":
    results = run_network_size_experiment()
    
    print("\n" + "="*50)
    print("NETWORK SIZE STUDY COMPLETED")
    print("="*50)
    print(f"Total networks tested: {len(results)}")
    print("\nBest performing networks:")
    
    best_density = min(results, key=lambda x: x['density_rmse'])
    best_gravity = min(results, key=lambda x: x['gravity_rmse'])
    best_r2 = max(results, key=lambda x: x['density_r2'])
    
    print(f"  Best Density RMSE: {best_density['name']} ({best_density['density_rmse']:.4f})")
    print(f"  Best Gravity RMSE: {best_gravity['name']} ({best_gravity['gravity_rmse']:.4f})")
    print(f"  Best R² Score: {best_r2['name']} ({best_r2['density_r2']:.3f})")
    
    print(f"\nAll results and figures saved in 'publication_figures/' directory")