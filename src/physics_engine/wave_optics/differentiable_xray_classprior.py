"""Recovering hidden geometry from a flow field with a SHAPE-CLASS PRIOR, through a differentiable lattice-
Boltzmann forward model.

Inverting for the geometry as a free per-cell FIELD is ill-posed (the flow does not determine thousands of cell
occupancies). The fix is a class prior: parameterise the unknown shape with a FEW parameters of a known class
(here a circle: cx, cy, r) instead of a dense field, which makes the problem massively over-determined and hence
well-posed. A smooth circle indicator theta = sigmoid((r^2 - d^2)/tau) keeps theta differentiable in (cx, cy, r),
so the adjoint carries the flow-field loss back through theta to the three parameters.

Forward model: D2Q9 lattice Boltzmann with a partial-bounce-back (grey) obstacle term weighted by theta, driven by
a body force, no-slip walls in y; the loss is the squared velocity-field mismatch against the reference flow.

GATES: (a) the recovered (cx, cy, r) matches the reference within 5%; (b) the adjoint gradient of the loss with
respect to the parameters matches central finite differences on r (a noise-robust instrument check).

I/O: no input files; prints the gradient check, the descent trace and a verdict. The reference field is generated
by the same forward model (a twin experiment), so nothing is fitted to external data. Uses warp; runs on CPU when
no CUDA device is present.
"""
import sys
import numpy as np
import warp as wp

wp.init()
DEV = "cuda:0" if wp.is_cuda_available() else "cpu"
CX = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1], dtype=np.float32)
CY = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1], dtype=np.float32)
WT = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=np.float32)
OP = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=np.int32)


@wp.kernel
def theta_circle(params: wp.array(dtype=wp.float32), tau: float, theta: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()                                       # smooth circle: theta=sigmoid((r^2-d^2)/tau), differentiable in (cx,cy,r)
    cx = params[0]; cy = params[1]; r = params[2]
    d2 = (float(i) - cx) * (float(i) - cx) + (float(j) - cy) * (float(j) - cy)
    z = wp.clamp((r * r - d2) / tau, -30.0, 30.0)         # clamped: no exp overflow in the backward pass (saturated sigmoid)
    theta[i, j] = 1.0 / (1.0 + wp.exp(-z))


@wp.kernel
def collide(f0: wp.array3d(dtype=wp.float32), fpost: wp.array3d(dtype=wp.float32),
            theta: wp.array2d(dtype=wp.float32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
            w: wp.array(dtype=wp.float32), omega: float, drive: float, alpha: float):
    i, j = wp.tid()
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        fk = f0[k, i, j]; rho += fk; mx += cx[k] * fk; my += cy[k] * fk
    uxr = mx / rho; uyr = my / rho; th = theta[i, j]
    denom = 1.0 + 0.5 * alpha * th
    ux = (uxr + 0.5 * drive) / denom; uy = uyr / denom
    Fx = (ux - uxr) * 2.0; Fy = (uy - uyr) * 2.0
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = cx[k] * ux + cy[k] * uy
        feq = w[k] * rho * (1.0 + 3.0 * cu + 4.5 * cu * cu - 1.5 * usq)
        Fk = (1.0 - 0.5 * omega) * 3.0 * w[k] * (cx[k] * Fx + cy[k] * Fy)
        fpost[k, i, j] = f0[k, i, j] - omega * (f0[k, i, j] - feq) + Fk


@wp.kernel
def stream(fpost: wp.array3d(dtype=wp.float32), f1: wp.array3d(dtype=wp.float32),
           solid: wp.array2d(dtype=wp.int32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
           opp: wp.array(dtype=wp.int32), nx: int, ny: int):
    i, j = wp.tid()
    for k in range(9):
        si = i - int(cx[k]); sj = j - int(cy[k])
        if si < 0: si += nx
        if si >= nx: si -= nx
        if sj < 0: sj = 0
        if sj >= ny: sj = ny - 1
        if solid[si, sj] == 1:
            f1[k, i, j] = fpost[opp[k], i, j]
        else:
            f1[k, i, j] = fpost[k, si, sj]


@wp.kernel
def field_loss(f: wp.array3d(dtype=wp.float32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
               uxt: wp.array2d(dtype=wp.float32), uyt: wp.array2d(dtype=wp.float32), L: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        fk = f[k, i, j]; rho += fk; mx += cx[k] * fk; my += cy[k] * fk
    dx = mx / rho - uxt[i, j]; dy = my / rho - uyt[i, j]
    wp.atomic_add(L, 0, dx * dx + dy * dy)


@wp.kernel
def macro(f: wp.array3d(dtype=wp.float32), cx: wp.array(dtype=wp.float32), cy: wp.array(dtype=wp.float32),
          uxo: wp.array2d(dtype=wp.float32), uyo: wp.array2d(dtype=wp.float32)):
    i, j = wp.tid()
    rho = float(0.0); mx = float(0.0); my = float(0.0)
    for k in range(9):
        fk = f[k, i, j]; rho += fk; mx += cx[k] * fk; my += cy[k] * fk
    uxo[i, j] = mx / rho; uyo[i, j] = my / rho


def _equil(rho, ux, uy):
    nx, ny = rho.shape; f = np.empty((9, nx, ny), np.float32)
    usq = ux * ux + uy * uy
    for k in range(9):
        cu = CX[k] * ux + CY[k] * uy
        f[k] = WT[k] * rho * (1 + 3 * cu + 4.5 * cu * cu - 1.5 * usq)
    return f


NX, NY, T, OMEGA, DRIVE, ALPHA, TAU = 70, 44, 90, 1.0, 6e-5, 1.6, 4.0
_cx = wp.array(CX, dtype=wp.float32, device=DEV); _cy = wp.array(CY, dtype=wp.float32, device=DEV)
_w = wp.array(WT, dtype=wp.float32, device=DEV); _opp = wp.array(OP, dtype=wp.int32, device=DEV)
_solid_np = np.zeros((NX, NY), np.int32); _solid_np[:, 0] = 1; _solid_np[:, -1] = 1
_solid = wp.array(_solid_np, dtype=wp.int32, device=DEV)
_f0 = _equil(np.ones((NX, NY), np.float32), np.zeros((NX, NY), np.float32), np.zeros((NX, NY), np.float32))


def forward(params_v, uxt=None, uyt=None, tape=None, want_field=False):
    rg = tape is not None
    pr = wp.array(np.array(params_v, np.float32), dtype=wp.float32, device=DEV, requires_grad=rg)
    th = wp.zeros((NX, NY), dtype=wp.float32, device=DEV, requires_grad=rg)
    cf = wp.array(_f0, dtype=wp.float32, device=DEV, requires_grad=rg)
    L = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=rg); keep = [cf]
    def run():
        wp.launch(theta_circle, (NX, NY), inputs=[pr, TAU, th], device=DEV)
        c = cf
        for t in range(T):
            fp = wp.zeros((9, NX, NY), dtype=wp.float32, device=DEV, requires_grad=rg)
            fn = wp.zeros((9, NX, NY), dtype=wp.float32, device=DEV, requires_grad=rg)
            wp.launch(collide, (NX, NY), inputs=[c, fp, th, _cx, _cy, _w, OMEGA, DRIVE, ALPHA], device=DEV)
            wp.launch(stream, (NX, NY), inputs=[fp, fn, _solid, _cx, _cy, _opp, NX, NY], device=DEV)
            keep.append(fp); keep.append(fn); c = fn
        if not want_field:
            wp.launch(field_loss, (NX, NY), inputs=[c, _cx, _cy, uxt, uyt, L], device=DEV)
        return c
    if want_field:
        c = run(); wp.synchronize()
        uxo = wp.zeros((NX, NY), dtype=wp.float32, device=DEV); uyo = wp.zeros((NX, NY), dtype=wp.float32, device=DEV)
        wp.launch(macro, (NX, NY), inputs=[c, _cx, _cy, uxo, uyo], device=DEV); wp.synchronize()
        return uxo.numpy(), uyo.numpy()
    if rg:
        with tape: run()
    else:
        run()
    wp.synchronize(); return L, pr


def main():
    print("=" * 80); print(f"HIDDEN-GEOMETRY RECOVERY WITH A SHAPE-CLASS PRIOR — circle from a flow field (device={DEV})"); print("=" * 80)
    true_p = [40.0, 20.0, 7.0]                              # the hidden circle (cx,cy,r), unknown to the inversion
    uxt_np, uyt_np = forward(true_p, want_field=True)
    uxt = wp.array(uxt_np, dtype=wp.float32, device=DEV); uyt = wp.array(uyt_np, dtype=wp.float32, device=DEV)
    print(f"\n  hidden geometry (cx,cy,r)={true_p}; class prior = circle (3 parameters instead of thousands of field cells => well-posed)")

    # (b) instrument: d(loss)/d(param) adjoint vs central finite difference on r
    test_p = [37.0, 22.0, 5.0]
    tape = wp.Tape(); L, pr = forward(test_p, uxt=uxt, uyt=uyt, tape=tape); tape.backward(loss=L)
    ad = pr.grad.numpy().copy()
    print(f"\n  (b) INSTRUMENT d(loss)/d(cx,cy,r) adjoint = {np.array2string(ad, precision=3)}; finite-difference check on r:")
    instr_ok = False
    for eps in [0.3, 0.1]:
        pp = test_p.copy(); pp[2] += eps; pm = test_p.copy(); pm[2] -= eps
        Lp = float(forward(pp, uxt=uxt, uyt=uyt)[0].numpy()[0]); Lm = float(forward(pm, uxt=uxt, uyt=uyt)[0].numpy()[0])
        fd = (Lp - Lm) / (2 * eps); rel = abs(ad[2] - fd) / (abs(fd) + 1e-12)
        if rel < 0.08: instr_ok = True
        print(f"      eps={eps}: d/dr finite-difference {fd:.3e} vs adjoint {ad[2]:.3e} -> rel {rel*100:.1f}% {'ok' if rel < 0.08 else ''}")

    # (a) recover (cx,cy,r) by gradient descent from a wrong initial guess
    p = np.array([35.0, 18.0, 5.0], np.float32); lr = np.array([8e3, 8e3, 8e3], np.float32)
    print(f"\n  (a) RECOVERY (start cx,cy,r={list(p)}):")
    print(f"  iter | cx     | cy     | r      | loss")
    for it in range(140):
        tape = wp.Tape(); L, pr = forward(list(p), uxt=uxt, uyt=uyt, tape=tape)
        Lv = float(L.numpy()[0]); tape.backward(loss=L); g = pr.grad.numpy()
        if it % 20 == 0 or it == 139:
            print(f"  {it:4d} | {p[0]:.3f} | {p[1]:.3f} | {p[2]:.3f} | {Lv:.3e}", flush=True)
        cap = 1.2 * (0.3 ** (it / 140.0))                 # shrinking trust region (1.2 -> 0.36): settles without oscillation
        step = np.clip(lr * g, -cap, cap); p = p - step
        p[0] = np.clip(p[0], 10, 60); p[1] = np.clip(p[1], 6, 38); p[2] = np.clip(p[2], 2, 14)
    errs = [abs(p[i] - true_p[i]) / true_p[i] for i in range(3)]
    calib_ok = max(errs) < 0.05
    print(f"\n  recovered (cx,cy,r)=({p[0]:.2f},{p[1]:.2f},{p[2]:.2f}) vs reference {true_p}; error cx {errs[0]*100:.1f}% cy {errs[1]*100:.1f}% r {errs[2]*100:.1f}%")

    all_ok = instr_ok and calib_ok
    print("\n" + "=" * 80)
    print(f"VERDICT: hidden-geometry recovery with a class prior = {'VALIDATED (both legs)' if all_ok else 'PARTIAL'} "
          f"(instrument adjoint=finite-difference {'ok' if instr_ok else 'no'} - geometry recovered <5% {'ok' if calib_ok else 'no'})")
    print(f"  Hidden geometry is recovered from the flow field alone through a class prior (3 parameters), which makes")
    print(f"  the inverse problem well-posed where a dense field inversion is not. Same adjoint inverse; the prior is")
    print(f"  what makes it identifiable. Scope: one shape class, one obstacle, noise-free reference field.")
    print("=" * 80)
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
