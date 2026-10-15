# Apply the sensitivity fixes to every example script (run once on the fix-sensitivity branch).
#   (1) prism primitive: one-argument arctan (atan2 adds a +-pi branch for cells whose top lies
#       above the receiver) and logarithms without cancellation for negative arguments
#   (2) G assembled in float64 and returned as float32
#   (3) receivers 1 m above the top of the model instead of inside the top layer
# usage (sciml env): python checks/apply_fix.py

import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

KERNEL = '''def {name}(x, y, z):
    # Plouff (1976) / Blakely (1995) prism primitive x log(y + r) + y log(x + r) - z arctan(x y / (z r)),
    # returned with a minus sign as before. The arctangent is the one-argument form: atan2(x y, z r)
    # adds a +-pi branch when z < 0, i.e. for a cell whose top lies above the receiver. The
    # logarithms avoid the cancellation in a + r for a < 0 by using log(s2) - log(r - a).
    r = torch.sqrt(x**2 + y**2 + z**2)
    one = torch.ones_like(r)

    def clog(c, a, s2):
        pos = torch.log(torch.where(a >= 0, a + r, one))
        neg = torch.where(s2 > 0, torch.log(torch.where(s2 > 0, s2, one)), torch.zeros_like(r)) - \\
            torch.log(torch.where(a < 0, r - a, one))
        return torch.where(c == 0, torch.zeros_like(r), c * torch.where(a >= 0, pos, neg))

    zsafe = torch.where(z == 0, one, z)
    t3 = torch.where(z == 0, torch.zeros_like(r), z * torch.atan(x * y / (zsafe * r)))
    return -(clog(x, y, x**2 + z**2) + clog(y, x, y**2 + z**2) - t3)
'''

# receivers: per script, the expression replaced and the new one (1 m above the top of the model)
RECEIVERS = {
    "001-EncodingComparison_smooth.py": ("np.zeros(xx_obs.size)", "np.full(xx_obs.size, -(DZ / 2.0 + 1.0))"),
    "001-EncodingComparisons_blocky.py": ("-np.ones(XX.size)", "np.full(XX.size, -(dz / 2.0 + 1.0))"),
    "002a-BlockNetworkSizeComparison.py": ("-np.ones(xx_obs.size)", "np.full(xx_obs.size, -(DZ / 2.0 + 1.0))"),
    "002b-BlockGridSizeComparison.py": ("-np.ones(xx_obs.size)", "np.full(xx_obs.size, -(dz / 2.0 + 1.0))"),
    "003-ImplictvsExplicitRegularisation.py": ("-np.ones(xx_obs.size)", "np.full(xx_obs.size, -(DZ / 2.0 + 1.0))"),
    "004-ModelEnsambles.py": ("-np.ones(xx_obs.size)", "np.full(xx_obs.size, -(DZ / 2.0 + 1.0))"),
    "005-NoiseSensitivity.py": ("-np.ones(xx_obs.size)", "np.full(xx_obs.size, -(DZ / 2.0 + 1.0))"),
}


def patch(fname, old_rx, new_rx):
    path = os.path.join(ROOT, fname)
    s = open(path, encoding="utf-8").read()
    # (1) kernel
    m = re.search(r"def (a_integral_torch|A_integral_torch)\(x, y, z\):\n(?:    .*\n|\s*\n)*?(?=\n*@torch|\n*def )", s)
    assert m, f"{fname}: kernel not found"
    s = s[:m.start()] + KERNEL.format(name=m.group(1)) + s[m.end():]
    # (2) float64 assembly
    m = re.search(r"def (construct_sensitivity_matrix_G\w*)\((cell_grid), (data_points), [^)]*\):\n", s)
    assert m, f"{fname}: constructor not found"
    s = s[:m.end()] + "    cell_grid, data_points = cell_grid.double(), data_points.double()   # assemble in float64\n" + s[m.end():]
    n = len(re.findall(r"return \((gamma \* a|Gamma \* A)\)\.to\(device\)", s))
    assert n == 1, f"{fname}: return line count {n}"
    s = re.sub(r"return \((gamma \* a|Gamma \* A)\)\.to\(device\)", r"return (\1).to(torch.float32).to(device)", s)
    # (3) receivers
    assert s.count(old_rx) == 1, f"{fname}: receiver expression count {s.count(old_rx)}"
    s = s.replace(old_rx, new_rx)
    open(path, "w", encoding="utf-8", newline="").write(s)
    print("patched", fname)


for f, (o, n) in RECEIVERS.items():
    patch(f, o, n)
