import numpy as np
import matplotlib.pyplot as plt
from discretize import TensorMesh
from simpeg.potential_fields import gravity
from simpeg import (
    maps,
    data,
    data_misfit,
    inverse_problem,
    regularization,
    optimization,
    directives,
    inversion,
)
from simpeg.utils import plot2Ddata, depth_weighting


# ----------------------------------------------------------
# Li & Oldenburg depth weighting
# ----------------------------------------------------------
def li_oldenburg_weight(z, z0=50.0, beta=2.0):
    """
    Li & Oldenburg depth weighting function.

    Parameters
    ----------
    z : array_like
        Depth (positive downward, in m).
    z0 : float
        Offset parameter.
    beta : float
        Exponent.

    Returns
    -------
    w : array_like
        Depth weights.
    """
    return 1.0 / (z + z0) ** beta


# ----------------------------------------------------------
# Build mesh and true model
# ----------------------------------------------------------
def build_model(nx=40, ny=40, nz=20, dx=25.0, dy=25.0, dz=25.0):
    # Mesh with z positive upward (SimPEG convention)
    hx = [(dx, nx)]
    hy = [(dy, ny)]
    hz = [(dz, nz)]
    mesh = TensorMesh([hx, hy, hz], x0=["C", "C", -hz[0][0] * nz])

    # True density contrast model
    model = np.zeros(mesh.nC)

    # Add a block anomaly
    block = (
        (mesh.gridCC[:, 0] > 250.0)
        & (mesh.gridCC[:, 0] < 750.0)
        & (mesh.gridCC[:, 1] > 250.0)
        & (mesh.gridCC[:, 1] < 750.0)
        & (mesh.gridCC[:, 2] > -250.0)
        & (mesh.gridCC[:, 2] < -500.0)
    )
    model[block] = 400.0  # kg/m^3

    return mesh, model


# ----------------------------------------------------------
# Forward simulate gravity data
# ----------------------------------------------------------
def simulate_data(mesh, model):
    # Observation grid at surface z=0
    xr = np.linspace(0, 1000, 21)
    yr = np.linspace(0, 1000, 21)
    X, Y = np.meshgrid(xr, yr)
    Z = np.zeros_like(X)

    receiver_locations = np.c_[X.ravel(), Y.ravel(), Z.ravel()]

    receiver_list = gravity.receivers.Point(receiver_locations, components="gz")
    source_field = gravity.sources.SourceField(receiver_list=[receiver_list])
    survey = gravity.survey.Survey(source_field)

    simulation = gravity.simulation.Simulation3DIntegral(
        survey=survey,
        mesh=mesh,
        rhoMap=maps.IdentityMap(nP=mesh.nC),
        engine="choclo",
    )

    data_object = data.Data(survey, dobs=simulation.dpred(model))
    return simulation, survey, data_object, receiver_locations


# ----------------------------------------------------------
# Run inversion with choice of depth weighting
# ----------------------------------------------------------
def run_inversion(mesh, simulation, data_object, use_li_oldenburg=True):
    dmis = data_misfit.L2DataMisfit(data=data_object, simulation=simulation)

    reg = regularization.WeightedLeastSquares(mesh, mapping=maps.IdentityMap(nP=mesh.nC))

    # Depth weighting
    if use_li_oldenburg:
        z_cc = -mesh.gridCC[:, 2]  # depth (positive down)
        w_depth = li_oldenburg_weight(z_cc, z0=50.0, beta=2.0)
        reg.set_weights(depth=w_depth)
    else:
        wr = depth_weighting(mesh, reference_locs=None)
        reg.set_weights(depth=wr)

    opt = optimization.ProjectedGNCG(
        maxIter=15, lower=0.0, upper=500.0, maxIterLS=20, maxIterCG=10, tolCG=1e-3
    )
    inv_prob = inverse_problem.BaseInvProblem(dmis, reg, opt)

    directives_list = [
        directives.BetaEstimate_ByEig(),
        directives.TargetMisfit(),
    ]

    inv = inversion.BaseInversion(inv_prob, directives_list)
    m0 = np.zeros(mesh.nC)
    recovered_model = inv.run(m0)

    return recovered_model, inv_prob


# ----------------------------------------------------------
# Plot results
# ----------------------------------------------------------
def plot_results(mesh, true_model, recovered_model, receiver_locations, data_object, inv_prob):
    nx, ny, nz = mesh.shape_cells

    rho_true_3d = true_model.reshape((nx, ny, nz), order="F")
    rho_inv_3d = recovered_model.reshape((nx, ny, nz), order="F")

    # Slices
    iz = nz // 2
    iy = ny // 2
    ix = nx // 2

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))

    extent_xy = [mesh.vectorCCx.min(), mesh.vectorCCx.max(),
                 mesh.vectorCCy.min(), mesh.vectorCCy.max()]
    extent_xz = [mesh.vectorCCx.min(), mesh.vectorCCx.max(),
                 -mesh.vectorCCz.max(), -mesh.vectorCCz.min()]
    extent_yz = [mesh.vectorCCy.min(), mesh.vectorCCy.max(),
                 -mesh.vectorCCz.max(), -mesh.vectorCCz.min()]

    # --- TRUE MODEL (no flip) ---
    axes[0, 0].imshow(rho_true_3d[:, :, iz].T, origin="lower", extent=extent_xy, cmap="viridis")
    axes[0, 0].set_title(f"True Δρ XY @ z={-mesh.vectorCCz[iz]:.0f} m")

    axes[0, 1].imshow(rho_true_3d[:, iy, :].T, origin="upper", extent=extent_xz,
                      aspect="auto", cmap="viridis")
    axes[0, 1].set_title(f"True Δρ XZ @ y={mesh.vectorCCy[iy]:.0f} m")

    axes[0, 2].imshow(rho_true_3d[ix, :, :].T, origin="upper", extent=extent_yz,
                      aspect="auto", cmap="viridis")
    axes[0, 2].set_title(f"True Δρ YZ @ x={mesh.vectorCCx[ix]:.0f} m")

    # --- INVERTED MODEL (flip z) ---
    axes[1, 0].imshow(rho_inv_3d[:, :, iz].T, origin="lower", extent=extent_xy, cmap="viridis")
    axes[1, 0].set_title(f"Inverted Δρ XY @ z={-mesh.vectorCCz[iz]:.0f} m")

    axes[1, 1].imshow(rho_inv_3d[:, iy, ::-1].T, origin="upper", extent=extent_xz,
                      aspect="auto", cmap="viridis")
    axes[1, 1].set_title(f"Inverted Δρ XZ @ y={mesh.vectorCCy[iy]:.0f} m")

    axes[1, 2].imshow(rho_inv_3d[ix, :, ::-1].T, origin="upper", extent=extent_yz,
                      aspect="auto", cmap="viridis")
    axes[1, 2].set_title(f"Inverted Δρ YZ @ x={mesh.vectorCCx[ix]:.0f} m")

    # --- DATA ---
    dobs = data_object.dobs
    dpred = inv_prob.dpred
    resid = dobs - dpred
    rms = np.sqrt(np.mean((resid) ** 2))

    plot2Ddata(receiver_locations, dobs, ax=axes[2, 0], contourOpts={"cmap": "RdBu_r"})
    axes[2, 0].set_title("Observed gz (mGal)")

    plot2Ddata(receiver_locations, dpred, ax=axes[2, 1], contourOpts={"cmap": "RdBu_r"})
    axes[2, 1].set_title("Predicted gz (mGal)")

    plot2Ddata(receiver_locations, resid, ax=axes[2, 2], contourOpts={"cmap": "RdBu_r"})
    axes[2, 2].set_title(f"Residual gz (RMS={rms:.3f} mGal)")

    plt.tight_layout()
    plt.show()


# ----------------------------------------------------------
# Main
# ----------------------------------------------------------
if __name__ == "__main__":
    mesh, true_model = build_model()
    simulation, survey, data_object, rx_locs = simulate_data(mesh, true_model)
    recovered_model, inv_prob = run_inversion(mesh, simulation, data_object,
                                              use_li_oldenburg=True)  # change to False for SimPEG weights
    plot_results(mesh, true_model, recovered_model, rx_locs, data_object, inv_prob)
