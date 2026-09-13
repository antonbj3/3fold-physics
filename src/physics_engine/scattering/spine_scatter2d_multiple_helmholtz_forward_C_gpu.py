"""Hybrid complex128 Helmholtz multiple scattering.

SciPy evaluates cylindrical special functions on the host; dense assembly, solve,
and angular field evaluation use torch on the requested device. CPU sibling is
unchanged. End-to-end timings must include special functions and transfers.
Run scripts/verify_scattering_gpu.py for the original five gates and full parity.
"""
import numpy as np
from scipy.special import jv, hankel1
import torch


def solve_MS(k, pos, a, theta_inc, M, device="cuda"):
    pos = np.asarray(pos, dtype=np.float64)
    if pos.ndim != 2 or pos.shape[1] != 2 or not len(pos):
        raise ValueError("positions must have shape (N, 2)")
    if not np.isfinite(pos).all() or not np.isfinite([k, a, theta_inc]).all():
        raise ValueError("finite geometry and frequency required")
    if k <= 0 or a <= 0 or not isinstance(M, int) or M < 0:
        raise ValueError("positive frequency/radius and nonnegative integer order required")
    nc = len(pos)
    ns = np.arange(-M, M + 1)
    nn = len(ns)
    delta = pos[:, None, :] - pos[None, :, :]
    radius = np.hypot(delta[..., 0], delta[..., 1])
    if np.any(radius[~np.eye(nc, dtype=bool)] <= 2 * a):
        raise ValueError("cylinders must be disjoint")
    np.fill_diagonal(radius, 1.)  # diagonal translation is removed below
    order = ns[None, :] - ns[:, None]
    # Host special functions are explicit: no claim of an entirely GPU path.
    distinct_orders = np.arange(-2 * M, 2 * M + 1)
    arguments, inverse = np.unique(k * radius, return_inverse=True)
    values = hankel1(distinct_orders[None, :], arguments[:, None])
    pair_values = values[inverse.reshape(nc, nc)]
    hankels = pair_values[:, :, order + 2 * M]
    s = torch.as_tensor(-jv(ns, k * a) / hankel1(ns, k * a), device=device)
    mode = torch.as_tensor(ns, dtype=torch.float64, device=device)
    angle = torch.as_tensor(np.arctan2(delta[..., 1], delta[..., 0]), device=device)
    diff = mode[None, :] - mode[:, None]
    blocks = -s[None, None, :, None] * torch.as_tensor(hankels, device=device)
    blocks = blocks * torch.exp(1j * diff[None, None] * angle[:, :, None, None])
    ids = torch.arange(nc, device=device)
    blocks[ids, ids] = torch.eye(nn, dtype=torch.complex128, device=device)
    matrix = blocks.permute(0, 2, 1, 3).reshape(nc * nn, nc * nn)
    positions = torch.as_tensor(pos, device=device)
    # Integer powers evaluated on host retain the original exact 4-cycle.
    powers = torch.as_tensor(1j ** ns, device=device)
    phase = torch.exp(1j * k * (np.cos(theta_inc) * positions[:, 0]
                              + np.sin(theta_inc) * positions[:, 1]))
    inc = phase[:, None] * powers * torch.exp(-1j * mode * theta_inc)
    rhs = (s * inc).reshape(-1)
    coefficients = torch.linalg.solve(matrix, rhs).reshape(nc, nn)
    return coefficients, mode, positions


def far_field(A, ns, pos, k, thetas):
    theta = torch.as_tensor(thetas, dtype=torch.float64, device=A.device)
    powers = torch.as_tensor((-1j) ** ns.detach().cpu().numpy(), device=A.device)
    harmonics = powers[None, :] * torch.exp(1j * theta[:, None] * ns[None, :])
    phase = torch.exp(-1j * k * (torch.cos(theta)[:, None] * pos[None, :, 0]
                               + torch.sin(theta)[:, None] * pos[None, :, 1]))
    return (phase * (harmonics @ A.T)).sum(dim=1)


def cross_sections(k, pos, a, theta_inc, M, nth=720, device="cuda"):
    A, ns, positions = solve_MS(k, pos, a, theta_inc, M, device)
    theta = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    f = far_field(A, ns, positions, k, theta)
    scat = (2. / (np.pi * k)) * torch.sum(torch.abs(f) ** 2) * (2 * np.pi / nth)
    ext = -(4. / k) * far_field(A, ns, positions, k, [theta_inc])[0].real
    return scat, ext, A, ns


def solve_MS_batch(k_values, pos, a, theta_inc, M, device="cuda"):
    """Solve independent frequencies with shared geometry and truncation order."""
    ks = np.asarray(k_values, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    if ks.ndim != 1 or not len(ks) or not np.isfinite(ks).all() or np.any(ks <= 0):
        raise ValueError("a nonempty vector of positive finite frequencies is required")
    if pos.ndim != 2 or pos.shape[1] != 2 or not len(pos) or not np.isfinite(pos).all():
        raise ValueError("finite positions must have shape (N, 2)")
    if not np.isfinite([a, theta_inc]).all() or a <= 0 or not isinstance(M, int) or M < 0:
        raise ValueError("positive radius, finite angle and nonnegative integer order required")
    nc = len(pos); nb = len(ks); ns = np.arange(-M, M + 1); nn = len(ns)
    delta = pos[:, None, :] - pos[None, :, :]
    radius = np.hypot(delta[..., 0], delta[..., 1])
    if np.any(radius[~np.eye(nc, dtype=bool)] <= 2 * a):
        raise ValueError("cylinders must be disjoint")
    np.fill_diagonal(radius, 1.)
    order = ns[None, :] - ns[:, None]
    arguments, inverse = np.unique(ks[:, None, None] * radius, return_inverse=True)
    values = hankel1(np.arange(-2 * M, 2 * M + 1)[None, :], arguments[:, None])
    pairs = values[inverse.reshape(nb, nc, nc)]
    hankels = pairs[:, :, :, order + 2 * M]
    s = torch.as_tensor(-jv(ns[None, :], ks[:, None] * a) / hankel1(ns[None, :], ks[:, None] * a), device=device)
    mode = torch.as_tensor(ns, dtype=torch.float64, device=device)
    angle = torch.as_tensor(np.arctan2(delta[..., 1], delta[..., 0]), device=device)
    diff = mode[None, :] - mode[:, None]
    blocks = -s[:, None, None, :, None] * torch.as_tensor(hankels, device=device)
    blocks = blocks * torch.exp(1j * diff[None, None] * angle[:, :, None, None])
    ids = torch.arange(nc, device=device)
    blocks[:, ids, ids] = torch.eye(nn, dtype=torch.complex128, device=device)
    matrix = blocks.permute(0, 1, 3, 2, 4).reshape(nb, nc * nn, nc * nn)
    positions = torch.as_tensor(pos, device=device)
    powers = torch.as_tensor(1j ** ns, device=device)
    phase = torch.exp(torch.as_tensor(1j * ks, device=device)[:, None]
                      * (np.cos(theta_inc) * positions[:, 0] + np.sin(theta_inc) * positions[:, 1]))
    inc = phase[:, :, None] * powers * torch.exp(-1j * mode * theta_inc)
    rhs = (s[:, None, :] * inc).reshape(nb, nc * nn, 1)
    coefficients = torch.linalg.solve(matrix, rhs).reshape(nb, nc, nn)
    return coefficients, mode, positions


def far_field_batch(A, ns, pos, k_values, thetas):
    """Evaluate each independent coefficient block on the same angular grid."""
    theta = torch.as_tensor(thetas, dtype=torch.float64, device=A.device)
    ks = np.asarray(k_values, dtype=np.float64)
    powers = torch.as_tensor((-1j) ** ns.detach().cpu().numpy(), device=A.device)
    harmonics = powers[None, :] * torch.exp(1j * theta[:, None] * ns[None, :])
    phase = torch.exp(torch.as_tensor(-1j * ks, device=A.device)[:, None, None]
                      * (torch.cos(theta)[:, None] * pos[None, :, 0]
                         + torch.sin(theta)[:, None] * pos[None, :, 1]))
    return (phase * torch.matmul(harmonics, A.transpose(-1, -2))).sum(dim=2)


def cross_sections_batch(k_values, pos, a, theta_inc, M, nth=720, device="cuda"):
    """Compute independent cross sections in one batch, including all setup."""
    ks = np.asarray(k_values, dtype=np.float64)
    A, ns, positions = solve_MS_batch(ks, pos, a, theta_inc, M, device)
    theta = np.linspace(0, 2 * np.pi, nth, endpoint=False)
    f = far_field_batch(A, ns, positions, ks, theta)
    scat = torch.as_tensor(2. / (np.pi * ks), device=device) * torch.sum(torch.abs(f) ** 2, dim=1) * (2 * np.pi / nth)
    ext = torch.as_tensor(-(4. / ks), device=device) * far_field_batch(A, ns, positions, ks, [theta_inc])[:, 0].real
    return scat, ext, A, ns
