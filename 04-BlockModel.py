import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# ---------------------------
# Utilities / setup
# ---------------------------
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

def idx_flat(i, j, k, Nx, Ny, Nz):
    return i*Ny*Nz + j*Nz + k

def build_grad_ops_sparse(Nx, Ny, Nz, dx, dy, dz, device):
    """
    Build first-order finite-difference gradient operators (forward differences)
    D_x, D_y, D_z as sparse COO tensors with shape (N_edges_dir, N_cells).
    """
    Ncells = Nx*Ny*Nz
    # X-direction
    rows_x, cols_x, vals_x = [], [], []
    row = 0
    for i in range(Nx-1):
        for j in range(Ny):
            for k in range(Nz):
                c0 = idx_flat(i,   j, k, Nx, Ny, Nz)
                c1 = idx_flat(i+1, j, k, Nx, Ny, Nz)
                rows_x += [row, row]
                cols_x += [c1, c0]
                vals_x += [ 1.0/dx, -1.0/dx]
                row += 1
    Dx = torch.sparse_coo_tensor(
        indices=torch.tensor([rows_x, cols_x], dtype=torch.long),
        values=torch.tensor(vals_x, dtype=torch.float32),
        size=((Nx-1)*Ny*Nz, Ncells),
        device=device
    ).coalesce()

    # Y-direction
    rows_y, cols_y, vals_y = [], [], []
    row = 0
    for i in range(Nx):
        for j in range(Ny-1):
            for k in range(Nz):
                c0 = idx_flat(i, j,   k, Nx, Ny, Nz)
                c1 = idx_flat(i, j+1, k, Nx, Ny, Nz)
                rows_y += [row, row]
                cols_y += [c1, c0]
                vals_y += [ 1.0/dy, -1.0/dy]
                row += 1
    Dy = torch.sparse_coo_tensor(
        indices=torch.tensor([rows_y, cols_y], dtype=torch.long),
        values=torch.tensor(vals_y, dtype=torch.float32),
        size=(Nx*(Ny-1)*Nz, Ncells),
        device=device
    ).coalesce()

    # Z-direction
    rows_z, cols_z, vals_z = [], [], []
    row = 0
    for i in range(Nx):
        for j in range(Ny):
            for k in range(Nz-1):
                c0 = idx_flat(i, j, k,   Nx, Ny, Nz)
                c1 = idx_flat(i, j, k+1, Nx, Ny, Nz)
                rows_z += [row, row]
                cols_z += [c1, c0]
                vals_z += [ 1.0/dz, -1.0/dz]
                row += 1
    Dz = torch.sparse_coo_tensor(
        indices=torch.tensor([rows_z, cols_z], dtype=torch.long),
        values=torch.tensor(vals_z, dtype=torch.float32),
        size=(Nx*Ny*(Nz-1), Ncells),
        device=device
    ).coalesce()

    return Dx, Dy, Dz

def depth_weights(grid_coords_tensor, z0, beta, Nx, Ny, Nz, normalize=True):
    """
    Li & Oldenburg-style depth weighting: w_i = 1 / (z_i + z0)^beta
    """
    z = grid_coords_tensor[:, 2]  # cell-center depth (>=0)
    w = 1.0 / torch.pow(z + z0, beta)
    if normalize:
        w = w / (w.mean() + 1e-12)
    return w  # length Ncells

def A_integral_torch(x, y, z):
    eps = 1e-20
    r = torch.sqrt(x**2 + y**2 + z**2).clamp_min(eps)
    return -(x * torch.log(torch.abs(y + r) + eps) +
             y * torch.log(torch.abs(x + r) + eps) -
             z * torch.atan2(x * y, z * r + eps))

@torch.inference_mode()
def construct_sensitivity_matrix_G_torch(cell_grid, data_points, d1, d2, device):
    """
    Full-tensor gravity vertical component kernel integrated over rectangular prisms.
    Returns dense G of shape (N_data, N_cells).
    """
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

# ---------------------------
# Conjugate gradient solver
# ---------------------------
def cg_solve(matvec, b, x0=None, max_iter=500, tol=1e-6, verbose=True):
    """
    Conjugate Gradient on SPD system A x = b with matvec closure.
    Returns x and history of ||r|| / ||b||.
    """
    x = torch.zeros_like(b) if x0 is None else x0.clone()
    r = b - matvec(x)
    p = r.clone()
    rs_old = torch.dot(r, r)
    bnorm = torch.sqrt(torch.dot(b, b) + 1e-30)
    hist = [float(torch.sqrt(rs_old)/bnorm)]
    if verbose:
        print(f"CG iter 0: rel_res = {hist[-1]:.3e}")
    for it in range(1, max_iter+1):
        Ap = matvec(p)
        pAp = torch.dot(p, Ap) + 1e-30
        alpha = rs_old / pAp
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r, r)
        rel_res = float(torch.sqrt(rs_new)/bnorm)
        hist.append(rel_res)
        if verbose and (it % 10 == 0 or rel_res < tol):
            print(f"CG iter {it:4d}: rel_res = {rel_res:.3e}")
        if rel_res < tol:
            break
        beta = rs_new / (rs_old + 1e-30)
        p = r + beta * p
        rs_old = rs_new
    return x, hist

# ---------------------------
# Main
# ---------------------------
def run():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs('plots', exist_ok=True)

    # Grid
    dx = dy = 50.0
    dz = 50.0
    x = np.arange(0.0, 1000.0 + dx, dx)
    y = np.arange(0.0, 1000.0 + dy, dy)
    z = np.arange(0.0, 500.0 + dz, dz)
    Nx, Ny, Nz = len(x), len(y), len(z)

    X3, Y3, Z3 = np.meshgrid(x.astype(float), y.astype(float), z.astype(float), indexing='ij')
    grid_coords = np.stack([X3.ravel(), Y3.ravel(), Z3.ravel()], axis=1)

    # Cell geometry for G
    dz_half = dz / 2.0
    cell_grid = np.hstack([grid_coords, np.full((grid_coords.shape[0], 1), dz_half)])
    cell_grid = torch.tensor(cell_grid, dtype=torch.float32, device=device)

    # Observation points (surface at z=-1 m to avoid singularity)
    XX, YY = np.meshgrid(x, y, indexing='ij')
    obs = np.column_stack([XX.ravel(), YY.ravel(), -np.ones(XX.size)])
    obs = torch.tensor(obs, dtype=torch.float32, device=device)

    # Sensitivity matrix
    print("Assembling sensitivity G ...")
    t0 = time.time()
    G = construct_sensitivity_matrix_G_torch(cell_grid, obs, dx, dy, device)
    G = G.clone().detach().requires_grad_(False)
    print(f"G shape = {tuple(G.shape)}, time = {time.time() - t0:.2f}s")

    # True model and data
    rho_true_vec, rho_true_3d = make_block_model(Nx, Ny, Nz, dx, dy, dz, rho_bg=0.0, rho_blk=400.0)
    rho_true_vec = rho_true_vec.to(device)
    with torch.no_grad():
        gz_true = (G @ rho_true_vec.unsqueeze(1)).squeeze(1)

    # Noisy data
    nl = 0.01
    sigma = nl * gz_true.std()
    noise = sigma * torch.randn_like(gz_true)
    gz_obs = gz_true + noise
    Wd = 1.0 / sigma  # scalar (data standard deviation)
    print(f"Noise std ≈ {float(sigma*1e5):.3f} mGal")

    # ---------------------------
    # Regularization setup
    # ---------------------------
    Ncells = Nx*Ny*Nz
    grid_coords_t = torch.tensor(grid_coords, dtype=torch.float32, device=device)

    # Depth weighting (Li & Oldenburg)
    reg_cfg = dict(
        z0=dz,      # stabilization depth (m)
        beta=1.5,   # gravity recommended ~1.5
        alpha_s=1e-2,  # smallness (depth-weighted)
        alpha_x=1,   # smoothness weights
        alpha_y=1,
        alpha_z=1
    )
    w_depth = depth_weights(grid_coords_t, reg_cfg['z0'], reg_cfg['beta'], Nx, Ny, Nz, normalize=True)
    w2 = w_depth**2  # used in smallness term

    # Sparse gradient operators
    Dx, Dy, Dz = build_grad_ops_sparse(Nx, Ny, Nz, dx, dy, dz, device)

    # ---------------------------
    # Linear system (whitened)
    # Minimize ||Wd(G m - d)||^2 + αs^2 ||W m||^2 + αx^2||Dx m||^2 + ...
    # ---------------------------
    Gw = Wd * G           # whitened sensitivity
    dw = Wd * gz_obs      # whitened data
    GT = Gw.transpose(0, 1).contiguous()

    def matvec(m_vec):
        # data term
        tmp = Gw @ m_vec
        Atmp = GT @ tmp
        # smallness with depth weighting
        reg_s = reg_cfg['alpha_s'] * reg_cfg['alpha_s'] * (w2 * m_vec)
        # smoothness
        reg_x = reg_cfg['alpha_x'] * reg_cfg['alpha_x'] * torch.sparse.mm(Dx.transpose(0,1), torch.sparse.mm(Dx, m_vec.unsqueeze(1))).squeeze(1)
        reg_y = reg_cfg['alpha_y'] * reg_cfg['alpha_y'] * torch.sparse.mm(Dy.transpose(0,1), torch.sparse.mm(Dy, m_vec.unsqueeze(1))).squeeze(1)
        reg_z = reg_cfg['alpha_z'] * reg_cfg['alpha_z'] * torch.sparse.mm(Dz.transpose(0,1), torch.sparse.mm(Dz, m_vec.unsqueeze(1))).squeeze(1)
        return Atmp + reg_s + reg_x + reg_y + reg_z

    b = GT @ dw  # m_ref = 0

    # Solve with CG
    print("Solving (CG) ...")
    m0 = torch.zeros(Ncells, dtype=torch.float32, device=device)
    m_inv, cg_hist = cg_solve(matvec, b, x0=m0, max_iter=800, tol=5e-5, verbose=True)

    # Predicted data
    with torch.no_grad():
        gz_pred = (G @ m_inv.unsqueeze(1)).squeeze(1)

    # ---------------------------
    # Plotting
    # ---------------------------
    def get_axes_coords():
        x1d = grid_coords[:, 0].reshape(Nx, Ny, Nz)[:, 0, 0]
        y1d = grid_coords[:, 1].reshape(Nx, Ny, Nz)[0, :, 0]
        z1d = grid_coords[:, 2].reshape(Nx, Ny, Nz)[0, 0, :]
        return x1d, y1d, z1d

    x1d, y1d, z1d = get_axes_coords()
    block_boundaries = get_block_boundaries(Nx, Ny, Nz)

    ix, iy, iz = Nx // 2, Ny // 2, min(Nz - 1, 5)
    tru = rho_true_3d.detach().cpu().numpy()
    inv = m_inv.view(Nx, Ny, Nz).detach().cpu().numpy()


    tru_max = 250
    inv_max = 250

    fig1, axes = plt.subplots(3, 3, figsize=(16, 15))

    # cell-edge limits
    x_edge_min, x_edge_max = x1d[0] - dx/2, x1d[-1] + dx/2
    y_edge_min, y_edge_max = y1d[0] - dy/2, y1d[-1] + dy/2
    z_edge_min, z_edge_max = z1d[0] - dz/2, z1d[-1] + dz/2

    extent_xy = [x_edge_min, x_edge_max, y_edge_min, y_edge_max]
    extent_xz = [x_edge_min, x_edge_max, z_edge_max, z_edge_min]
    extent_yz = [y_edge_min, y_edge_max, z_edge_max, z_edge_min]

    im = axes[0, 0].imshow(tru[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 0].set_title(f"True Δρ XY @ z≈{z1d[iz]:.0f} m")
    fig1.colorbar(im, ax=axes[0, 0], label='kg/m³', fraction=0.046, pad=0.04)
    im = axes[0, 1].imshow(tru[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 1].set_title(f"True Δρ XZ @ y≈{y1d[iy]:.0f} m")
    im = axes[0, 2].imshow(tru[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=tru_max, cmap='viridis')
    axes[0, 2].set_title(f"True Δρ YZ @ x≈{x1d[ix]:.0f} m")

    im = axes[1, 0].imshow(inv[:, :, iz].T, origin='lower', extent=extent_xy, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 0].set_title(f"L2+Smooth Δρ XY @ z≈{z1d[iz]:.0f} m")
    boundary_for_z = next((b for b in block_boundaries if b[4] == iz), None)
    if boundary_for_z:
        xs, xe, ys, ye, _ = boundary_for_z
        rect = plt.Rectangle((x[xs] - dx/2, y[ys] - dy/2), (xe - xs) * dx, (ye - ys) * dy, edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 0].add_patch(rect)
    fig1.colorbar(im, ax=axes[1, 0], label='kg/m³', fraction=0.046, pad=0.04)

    im = axes[1, 1].imshow(inv[:, iy, :].T, origin='upper', extent=extent_xz, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 1].set_title(f"L2+Smooth Δρ XZ @ y≈{y1d[iy]:.0f} m")
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
        rect = plt.Rectangle((x[xs] - dx/2, z[min_z_idx] - dz/2), (xe - xs) * dx, (max_z_idx - min_z_idx + 1) * dz,
                             edgecolor='white', facecolor='none', linewidth=2)
        axes[1, 1].add_patch(rect)

    im = axes[1, 2].imshow(inv[ix, :, :].T, origin='upper', extent=extent_yz, aspect='auto', vmin=0, vmax=inv_max, cmap='viridis')
    axes[1, 2].set_title(f"L2+Smooth Δρ YZ @ x≈{x1d[ix]:.0f} m")
    for xs, xe, ys, ye, z_idx in block_boundaries:
        if xs <= ix < xe:
            rect = plt.Rectangle((y[ys] - dy/2, z[z_idx] - dz/2), (ye - ys) * dy, dz, edgecolor='white', facecolor='none', linewidth=2)
            axes[1, 2].add_patch(rect)

    for ax in [axes[0, 1], axes[0, 2], axes[1, 1], axes[1, 2]]:
        ax.set_aspect(1.0)
    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_aspect(1.0)

    for ax in [axes[0, 0], axes[1, 0]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
    for ax in [axes[0, 1], axes[1, 1]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Depth (m)')
    for ax in [axes[0, 2], axes[1, 2]]:
        ax.set_xlabel('y (m)')
        ax.set_ylabel('Depth (m)')

    def to_mgal(g):
        return 1e5 * g.detach().cpu().numpy()

    obs_mgal = to_mgal(gz_obs)
    pre_mgal = to_mgal(gz_pred)
    res_mgal = obs_mgal - pre_mgal

    obs_x = obs[:, 0].cpu().numpy()
    obs_y = obs[:, 1].cpu().numpy()

    v = max(abs(obs_mgal).max(), abs(pre_mgal).max())

    sc = axes[2, 0].scatter(obs_x, obs_y, c=obs_mgal, s=80, cmap='viridis', vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 0].set_title('Observed gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 0], fraction=0.046, pad=0.04)

    sc = axes[2, 1].scatter(obs_x, obs_y, c=pre_mgal, s=80, cmap='viridis', vmin=-v, vmax=v, marker='o', edgecolors='none')
    axes[2, 1].set_title('Predicted gz (mGal)')
    fig1.colorbar(sc, ax=axes[2, 1], fraction=0.046, pad=0.04)

    vmax_res = np.abs(res_mgal).max()
    sc = axes[2, 2].scatter(obs_x, obs_y, c=res_mgal, s=80, cmap='viridis', vmin=-vmax_res, vmax=vmax_res, marker='o', edgecolors='none')
    axes[2, 2].set_title(f'Residual (RMS={np.sqrt(np.mean(res_mgal**2)):.3f} mGal)')
    fig1.colorbar(sc, ax=axes[2, 2], fraction=0.046, pad=0.04)

    for ax in [axes[2, 0], axes[2, 1], axes[2, 2]]:
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')
        ax.set_aspect('equal')

    fig1.tight_layout()
    fig1.tight_layout(rect=[0, 0.05, 1, 1])
    fig1.savefig('plots/DeterministicBlockModel.png', dpi=300)
    plt.close(fig1)

    # CG convergence history
    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.plot(cg_hist, color='black')
    ax.set_yscale('log')
    ax.grid(True, which='both', ls='--', alpha=0.3)
    ax.set_xlabel('CG iteration')
    ax.set_ylabel('Relative residual ||r||/||b||')
    fig3.tight_layout()
    fig3.savefig('plots/DeterministicBlockModel_convergence.png', dpi=300)
    plt.close(fig3)

    # Metrics
    rms_rho = torch.sqrt(torch.mean((m_inv - rho_true_vec.to(device)) ** 2)).item()
    rms_gz = torch.sqrt(torch.mean((gz_pred - gz_obs) ** 2)).item() * 1e5
    print(f"RMS density-contrast error ≈ {rms_rho:.2f} kg/m^3")
    print(f"RMS data misfit ≈ {rms_gz:.3f} mGal")

if __name__ == '__main__':
    run()
