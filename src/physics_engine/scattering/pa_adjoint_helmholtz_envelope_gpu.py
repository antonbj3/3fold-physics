"""Torch float64 separable solve of the unchanged five-point FD operator.

The CPU operator has identity boundary rows but interior-to-boundary columns:
L is NOT symmetric. Its transpose solution therefore needs nonzero boundary
values reconstructed from adjacent interior values. Sine diagonalization solves
exactly the same interior stencil, including indefinite Helmholtz frequencies.
No continuum Green function or positive-definite iterative approximation is used.
"""
import math
import torch


class HelmholtzOperator:
    def __init__(self, n, h, kappa=0., device="cuda"):
        if not isinstance(n, int) or n < 3 or not math.isfinite(h) or h <= 0:
            raise ValueError("integer n >= 3 and positive finite spacing required")
        if not math.isfinite(kappa):
            raise ValueError("finite kappa required")
        self.n, self.h, self.kappa = n, h, kappa
        self.device = torch.device(device)
        modes = torch.arange(1, n - 1, dtype=torch.float64, device=device)
        self.basis = math.sqrt(2 / (n - 1)) * torch.sin(math.pi / (n - 1)
                                                     * modes[:, None] * modes[None, :])
        lam = 4 / h ** 2 * torch.sin(math.pi * modes / (2 * (n - 1))) ** 2
        self.eigenvalues = lam[:, None] + lam[None, :] - kappa ** 2
        if torch.any(torch.abs(self.eigenvalues) <= 64 * torch.finfo(torch.float64).eps
                     * (8 / h ** 2 + kappa ** 2)):
            raise ValueError("singular or numerically unresolved interior operator")

    def solve(self, rhs, adjoint=False):
        rhs = torch.as_tensor(rhs, dtype=torch.float64, device=self.device)
        if rhs.shape != (self.n, self.n) or not torch.isfinite(rhs).all():
            raise ValueError("finite (n, n) right-hand side required")
        interior = rhs[1:-1, 1:-1].clone()
        d = 1 / self.h ** 2
        if not adjoint:
            interior[0] += d * rhs[0, 1:-1]
            interior[-1] += d * rhs[-1, 1:-1]
            interior[:, 0] += d * rhs[1:-1, 0]
            interior[:, -1] += d * rhs[1:-1, -1]
        q = self.basis
        z = rhs.clone()
        z[1:-1, 1:-1] = q @ ((q.T @ interior @ q) / self.eigenvalues) @ q.T
        if adjoint:
            z[0, 1:-1] += d * z[1, 1:-1]
            z[-1, 1:-1] += d * z[-2, 1:-1]
            z[1:-1, 0] += d * z[1:-1, 1]
            z[1:-1, -1] += d * z[1:-1, -2]
        return z


def solve_adj(n=201, h=.005, kappa=0., src=(100, 100), device="cuda"):
    if len(src) != 2 or any(not isinstance(i, int) or i < 0 or i >= n for i in src):
        raise ValueError("source must be an in-grid integer pair")
    operator = HelmholtzOperator(n, h, kappa, device)
    rhs = torch.zeros((n, n), dtype=torch.float64, device=device)
    rhs[src] = 1.
    return operator.solve(rhs, adjoint=True)
