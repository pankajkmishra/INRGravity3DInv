# Verify the gravity sensitivity matrices built by the example scripts.
#
# For every script, G is assembled exactly as the script does it (its own
# construct_sensitivity_matrix_G function, float32 tensors, its own grid and receivers) and
# compared with an independent float64 reference on the SAME geometry:
#   reference  Plouff prism formula in float64 with the one-argument arctan of Blakely (1995)
#              and cancellation-free logarithms,
#   check      the reference itself is checked against direct numerical integration
#              (scipy dblquad of 1/R(z1) - 1/R(z2) over the cell's x-y footprint, which is smooth
#              even when the receiver lies inside the cell), and against symmetry: a receiver at
#              the centre of a cell feels exactly zero vertical gravity from it.
# Separate error sources are reported: (1) the two-argument atan2 branch, (2) float32 assembly.
# The script also reports where the cells sit relative to the receivers (geometry).
#
# usage (sciml env): python checks/verify_sensitivity.py

import importlib.util
import os
import sys

import numpy as np
import torch
from scipy.integrate import dblquad

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GAMMA = 6.67430e-11


def load(name):
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), os.path.join(ROOT, name))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------- independent float64 reference ----------

def _logplus(a, s2, r):
    # log(a + r) with r^2 = a^2 + s2, without cancellation for a < 0
    with np.errstate(divide="ignore", invalid="ignore"):
        pos = np.log(np.where(a >= 0, a + r, 1.0))
        neg = np.where(s2 > 0, np.log(np.where(s2 > 0, s2, 1.0)), 0.0) - np.log(np.where(a < 0, r - a, 1.0))
    return np.where(a >= 0, pos, neg)


def _primitive(x, y, z):
    # x log(y + r) + y log(x + r) - z arctan(x y / (z r)); each term is zero when its factor is
    r = np.sqrt(x * x + y * y + z * z)
    t1 = np.where(x == 0, 0.0, x * _logplus(y, x * x + z * z, r))
    t2 = np.where(y == 0, 0.0, y * _logplus(x, y * y + z * z, r))
    with np.errstate(divide="ignore", invalid="ignore"):
        t3 = np.where(z == 0, 0.0, z * np.arctan(x * y / np.where(z == 0, 1.0, z * r)))
    return t1 + t2 - t3


def reference_G(cell_grid, obs, d1, d2):
    # float64, same corner convention as the scripts: G = GAMMA * sum mu * (-primitive)
    c = np.asarray(cell_grid, dtype=np.float64)
    o = np.asarray(obs, dtype=np.float64)
    G = np.empty((o.shape[0], c.shape[0]))
    for s in range(0, o.shape[0], 64):
        oo = o[s:s + 64]
        xs = [(c[None, :, 0] - d1 / 2) - oo[:, None, 0], (c[None, :, 0] + d1 / 2) - oo[:, None, 0]]
        ys = [(c[None, :, 1] - d2 / 2) - oo[:, None, 1], (c[None, :, 1] + d2 / 2) - oo[:, None, 1]]
        zs = [(c[None, :, 2] - c[None, :, 3]) - oo[:, None, 2], (c[None, :, 2] + c[None, :, 3]) - oo[:, None, 2]]
        acc = np.zeros((oo.shape[0], c.shape[0]))
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # index 1 = upper bound; the scripts add A(x2, y2, z2) with +, so mu = +1 when
                    # i + j + k is odd (an even number of lower bounds); A = -primitive
                    mu = 1.0 if (i + j + k) % 2 == 1 else -1.0
                    acc -= mu * _primitive(xs[i], ys[j], zs[k])
        G[s:s + 64] = GAMMA * acc
    return G


def quad_G(cell, o, d1, d2):
    # direct integration: int_z (z'-z0)/R^3 dz' = 1/R(z1) - 1/R(z2), then dblquad over x, y
    cx, cy, cz, czh = cell
    z1, z2 = cz - czh - o[2], cz + czh - o[2]

    def f(y, x):
        rh2 = x * x + y * y
        return 1.0 / np.sqrt(rh2 + z1 * z1) - 1.0 / np.sqrt(rh2 + z2 * z2)

    val, _ = dblquad(f, cx - d1 / 2 - o[0], cx + d1 / 2 - o[0], cy - d2 / 2 - o[1], cy + d2 / 2 - o[1],
                     epsabs=0.0, epsrel=1e-11)
    return GAMMA * val


# ---------- geometry of each script (copied from its run()/solve_case()) ----------

def geometry_block(dx, dy, dz, x_max, y_max, z_max):
    x = np.arange(0.0, x_max + dx, dx)
    y = np.arange(0.0, y_max + dy, dy)
    z = np.arange(0.0, z_max + dz, dz)
    x3, y3, z3 = np.meshgrid(x, y, z, indexing="ij")
    grid = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=1)
    cell = np.hstack([grid, np.full((grid.shape[0], 1), dz / 2.0)])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    obs = np.column_stack([xx.ravel(), yy.ravel(), -np.ones(xx.size)])
    return cell, obs, (len(x), len(y), len(z))


def geometry_smooth(m):
    x = np.linspace(0.0, (m.NX - 1) * m.DX, m.NX)
    y = np.linspace(0.0, (m.NY - 1) * m.DY, m.NY)
    z = np.linspace(0.0, (m.NZ - 1) * m.DZ, m.NZ)
    x3, y3, z3 = np.meshgrid(x, y, z, indexing="ij")
    grid = np.stack([x3.ravel(), y3.ravel(), z3.ravel()], axis=1)
    cell = np.hstack([grid, np.full((grid.shape[0], 1), m.DZ / 2.0)])
    xx, yy = np.meshgrid(x, y, indexing="ij")
    obs = np.column_stack([xx.ravel(), yy.ravel(), np.zeros(xx.size)])
    return cell, obs, (m.NX, m.NY, m.NZ)


def cases():
    out = []
    m = load("001-EncodingComparison_smooth.py")
    out.append(("001 smooth", m, m.construct_sensitivity_matrix_G, geometry_smooth(m), m.DX, m.DY))
    m = load("001-EncodingComparisons_blocky.py")
    out.append(("001 blocky", m, lambda c, o, a, b, d, m=m: m.construct_sensitivity_matrix_G_torch(c, o, a, b, d),
                geometry_block(m.DX, m.DY, m.DZ, m.X_MAX, m.Y_MAX, m.Z_MAX), m.DX, m.DY))
    for name, label in (("002a-BlockNetworkSizeComparison.py", "002a network size"),
                        ("003-ImplictvsExplicitRegularisation.py", "003 implicit vs explicit"),
                        ("004-ModelEnsambles.py", "004 ensembles"),
                        ("005-NoiseSensitivity.py", "005 noise")):
        m = load(name)
        out.append((label, m, m.construct_sensitivity_matrix_G,
                    geometry_block(m.DX, m.DY, m.DZ, m.X_MAX, m.Y_MAX, m.Z_MAX), m.DX, m.DY))
    m = load("002b-BlockGridSizeComparison.py")
    for lab, cs in m.GRID_CONFIGS.items():
        out.append((f"002b grid {lab}", m, m.construct_sensitivity_matrix_G,
                    geometry_block(cs, cs, cs, m.X_MAX, m.Y_MAX, m.Z_MAX), cs, cs))
    return out


def summary(label, G_test, G_ref, n_xyz, cell, obs):
    nx, ny, nz = n_xyz
    err = np.abs(G_test - G_ref)
    rel = err / np.abs(G_ref).max()           # relative to the largest entry of G
    # the cell directly below each receiver in the top layer (index z = 0)
    own = np.array([np.argmin((cell[:, 0] - o[0]) ** 2 + (cell[:, 1] - o[1]) ** 2 + (cell[:, 2] - cell[:, 2].min()) ** 2)
                    for o in obs])
    rows = np.arange(len(obs))
    own_err = np.abs(G_test[rows, own] - G_ref[rows, own]) / np.abs(G_ref).max()
    return (f"{label:26s} |dG|/max|G|: max {rel.max():.2e}, median {np.median(rel):.2e}; "
            f"own top cell (receiver inside): median {np.median(own_err):.2e}")


def main():
    torch.set_grad_enabled(False)
    print("Geometry: every script centres its cells on the grid nodes, so the top layer straddles the")
    print("surface; receivers are at z = -1 m (block scripts) or z = 0 (smooth script).\n")
    for label, mod, builder, (cell, obs, n_xyz), d1, d2 in cases():
        top = cell[:, 2].min() - cell[0, 3]
        print(f"{label}: cells {n_xyz}, top of the top layer at z = {top:+.0f} m, receivers at z = {obs[0, 2]:+.0f} m, "
              f"receivers inside a top cell: {top < obs[0, 2]}")
        G32 = builder(torch.tensor(cell, dtype=torch.float32), torch.tensor(obs, dtype=torch.float32), d1, d2, "cpu").double().numpy()
        G64s = builder(torch.tensor(cell, dtype=torch.float64), torch.tensor(obs, dtype=torch.float64), d1, d2, "cpu").double().numpy()
        Gref = reference_G(cell, obs, d1, d2)
        print("   " + summary("script as run (float32)", G32, Gref, n_xyz, cell, obs))
        print("   " + summary("script formula, float64", G64s, Gref, n_xyz, cell, obs))
        # independent check of the reference: direct integration on a few entries, including the
        # receiver's own top cell (inside it) and cells at increasing distance
        rng = np.random.default_rng(0)
        r0 = len(obs) // 2
        own = int(np.argmin(np.sum((cell[:, :2] - obs[r0, :2]) ** 2, axis=1) + (cell[:, 2] - cell[:, 2].min()) ** 2))
        cols = [own] + list(rng.choice(cell.shape[0], 5, replace=False))
        q = [quad_G(cell[c], obs[r0], d1, d2) for c in cols]
        worst = max(abs(Gref[r0, c] - v) / max(abs(v), 1e-30 + abs(Gref[r0]).max() * 1e-12) for c, v in zip(cols, q))
        print(f"   reference vs direct integration (6 entries incl. own cell): max rel. diff = {worst:.1e}; "
              f"own cell: ref {Gref[r0, own]:.4e}, quad {q[0]:.4e}, script {G32[r0, own]:.4e}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
