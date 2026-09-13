"""Gradient inverse design on the differentiable wave substrate: design an acoustic focusing lens.

The per-cell material field c^2(x) is optimised by bounded gradient ASCENT (c^2 clipped to physical speeds) so
that an incoming pulse concentrates its energy in a central target window. Each step costs ONE adjoint pass (a
full-field gradient), instead of the N^2 forward solves a finite-difference gradient would need.

Gate: the focus energy J rises substantially and monotonically, measured against the BEST UNIFORM medium rather
than the uniform-1.0 starting point (the start is a near-worst focuser, so a start-relative gain is inflated).

I/O: no input files; prints the iteration trace, the honest gain against the best uniform baseline and a verdict.
Scope: 2-D, lossless, c^2 only (no density or anisotropy), plain gradient ascent (no manufacturability
constraint). Uses warp; runs on CPU when no CUDA device is present.
"""
# --- sibling-package bootstrap: the source tree keeps these modules in one flat directory; this repository
# --- splits them by domain, so put every package directory on sys.path when run as a script.
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in sorted(_os.listdir(_PKG_ROOT)):
    _d = _os.path.join(_PKG_ROOT, _p)
    if _os.path.isdir(_d) and not _p.startswith('__') and _d not in _sys.path:
        _sys.path.insert(0, _d)
del _os, _sys, _p, _d
import sys
import numpy as np
import warp as wp
from diff_wave_substrate import forward, N, DEV

CLO, CHI = 0.25, 4.0      # c² bounds (speed 0.5..2.0)


def main():
    print("=" * 76)
    print(f"INVERSE-DESIGN (acoustic lens) on the differentiable wave substrate — device={DEV}")
    print("=" * 76)
    c = np.full((N, N), 1.0, np.float32)
    Js = []
    lr = 0.15
    for it in range(36):
        csq = wp.array(c, dtype=wp.float32, device=DEV, requires_grad=True)
        tape = wp.Tape()
        with tape:
            loss, _ = forward(csq)
        tape.backward(loss=loss)
        g = csq.grad.numpy()
        J = float(loss.numpy()[0]); Js.append(J)
        c = np.clip(c + lr * g / (np.abs(g).max() + 1e-12), CLO, CHI).astype(np.float32)  # bounded gradient ASCENT
        if it % 6 == 0 or it == 35:
            print(f"  iter {it:>2}: focus-energy J = {J:.4e}")
    J0, Jf = Js[0], Js[-1]
    monotone = sum(Js[i + 1] >= Js[i] - 1e-9 for i in range(len(Js) - 1)) / (len(Js) - 1)
    gain = Jf / (J0 + 1e-30)
    # HONEST baseline (audit fix): the start-relative gain is reference-inflated (uniform-1.0 is a near-worst focuser).
    # Compare to the BEST uniform medium = the fair baseline a non-designer would pick.
    Jus = []
    for cu in [0.4, 0.6, 0.8, 1.0, 1.4, 2.0, 2.8]:     # c<=1.67 => CFL-stable at the fixed dt=0.4*dx (cu=4 blows up)
        ju = float(forward(wp.array(np.full((N, N), cu, np.float32), dtype=wp.float32, device=DEV))[0].numpy()[0])
        if np.isfinite(ju):
            Jus.append(ju)
    Jbest = max(Jus)
    honest = Jf / (Jbest + 1e-30)
    hi = float((c > 2.0).mean()); lo = float((c < 0.5).mean())
    print(f"\n  J_start(uniform1.0)={J0:.3e} → J_final={Jf:.3e}; best-uniform baseline={Jbest:.3e}")
    print(f"  HONEST focus gain vs BEST uniform = {honest:.1f}×  (the {gain:.0f}× vs the uniform-1.0 START is reference-inflated)")
    print(f"  designed c²: {hi:.0%} >2, {lo:.0%} <0.5 — structured lens, monotone {monotone:.0%}")
    ok = honest > 3.0 and monotone > 0.8
    print("\n" + "=" * 76)
    print(f"VERDICT: gradient inverse-design = {'VALIDATED' if ok else 'PARTIAL'} ({honest:.1f}× vs best-uniform, {monotone:.0%} monotone)")
    print(f"  The substrate designs a focusing material through the adjoint (1 backward pass = full-field gradient).")
    print(f"  The same solver gives an EM lens or an elastic metamaterial by a coefficient swap. Scope: 2-D, lossless,")
    print(f"  c^2 only (no density or anisotropy), gradient ascent (no manufacturability constraint).")
    print("=" * 76)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
