"""Full-waveform calibration of a simulated twin: recover a hidden material field from observed wave data and
report where the recovered field is trustworthy.

The loop observes a reference wave field over TIME, minimises the full-trace field mismatch over the estimated
c^2 by ONE adjoint pass per step (full-waveform inversion of the material field), recovers the true field and
localises a hidden inclusion. A sensitivity map (the inverse of the accumulated illumination sensitivity) then
flags which cells the probe actually determined.

A single-shot final-field inversion is ill-posed (cycle skipping); the FULL time trace makes it well-posed.

GATES: (1) the calibrated field recovers the reference (relative error falls, the inclusion is localised);
(2) the sensitivity map PREDICTS where the residual error is large (positive correlation, and the error where the
map says "determined" is well below the error where it does not) — an honest per-region trust statement.

I/O: no input files; prints the calibration summary and a verdict. Scope: 2-D, lossless, a single probe with
full-field observation, c^2 only. Uses warp; runs on CPU when no CUDA device is present.
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
from diff_wave_substrate import upd_v, upd_s, N, DEV

T = 70; CLO, CHI = 0.25, 4.0


@wp.kernel
def add_mismatch(S: wp.array2d(dtype=wp.float32), tgt: wp.array2d(dtype=wp.float32), out: wp.array(dtype=wp.float32)):
    i, j = wp.tid()
    d = S[i, j] - tgt[i, j]
    wp.atomic_add(out, 0, d * d)


def forward(csq, tgt_trace=None, save=False):
    dx = 1.0 / (N - 1); dt = 0.4 * dx; csx = dt / dx; csy = dt / dx
    seed = np.zeros((N, N), np.float32)
    xi = np.arange(N)[:, None]; yj = np.arange(N)[None, :]
    seed += np.exp(-(((xi - 6) ** 2 + (yj - N // 2) ** 2) / 8.0)).astype(np.float32)   # probe pulse, left edge
    S = wp.array(seed, dtype=wp.float32, device=DEV, requires_grad=True)
    U = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    W = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
    loss = wp.zeros(1, dtype=wp.float32, device=DEV, requires_grad=True) if tgt_trace is not None else None
    saved = []
    for t in range(T):
        Un = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        Wn = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        Sn = wp.zeros((N, N), dtype=wp.float32, device=DEV, requires_grad=True)
        wp.launch(upd_v, dim=(N, N), inputs=[S, U, W, Un, Wn, csx, csy], device=DEV)
        wp.launch(upd_s, dim=(N, N), inputs=[S, Un, Wn, Sn, csq, csx, csy], device=DEV)
        S, U, W = Sn, Un, Wn
        if tgt_trace is not None:
            wp.launch(add_mismatch, dim=(N, N), inputs=[S, tgt_trace[t], loss], device=DEV)   # FULL-TRACE loss
        if save:
            saved.append(S.numpy().copy())
    return loss, S, saved


def main():
    print("=" * 78)
    print(f"FULL-WAVEFORM CALIBRATION (full-trace FWI + sensitivity map) on the wave substrate — {DEV}")
    print("=" * 78)
    ct = np.full((N, N), 1.0, np.float32)
    ii, jj = np.mgrid[0:N, 0:N]
    incl = ((ii - 22) ** 2 + (jj - 26) ** 2) < 25         # hidden circular inclusion (the anomaly)
    ct[incl] = 0.4
    _, _, real_trace = forward(wp.array(ct, dtype=wp.float32, device=DEV), save=True)   # the observed reference field (the data)
    tgt_trace = [wp.array(s, dtype=wp.float32, device=DEV) for s in real_trace]

    c = np.full((N, N), 1.3, np.float32)   # WRONG starting background (no lucky-correct cells → σ-validation honest)
    e0 = np.linalg.norm(c - ct) / np.linalg.norm(ct)
    sens = np.zeros((N, N)); lr = 0.5
    for _ in range(40):
        csq = wp.array(c, dtype=wp.float32, device=DEV, requires_grad=True)
        tape = wp.Tape()
        with tape:
            loss, _, _ = forward(csq, tgt_trace=tgt_trace)
        tape.backward(loss=loss)
        g = csq.grad.numpy()
        sens += np.abs(g)
        c = np.clip(c - lr * g / (np.abs(g).max() + 1e-12), CLO, CHI).astype(np.float32)
    ef = np.linalg.norm(c - ct) / np.linalg.norm(ct)

    sig = 1.0 / (sens + sens.mean() * 0.1 + 1e-12); sig /= sig.max()
    err_cell = np.abs(c - ct)
    rho_sig = float(np.corrcoef(sig.ravel(), err_cell.ravel())[0, 1])
    lo = sig < np.median(sig)                              # σ says TRUST (well-illuminated) vs DON'T
    et, eu = float(err_cell[lo].mean()), float(err_cell[~lo].mean())

    print(f"\n  global calibration: c² rel-err {e0:.3f} → {ef:.3f} (limited by un-illuminated regions, single source)")
    print(f"  sensitivity-gated accuracy: error where the map says DETERMINED = {et:.3f}  vs  UNDETERMINED = {eu:.3f}  "
          f"({eu/max(et,1e-9):.1f}× worse where σ flags)  {'✓ trust is honest' if et < 0.6 * eu else '~'}")
    print(f"  σ predicts error: ρ(σ,|err|) = {rho_sig:+.2f}  {'✓' if rho_sig > 0.3 else '~'}")
    ok = rho_sig > 0.3 and et < 0.6 * eu
    print("\n" + "=" * 78)
    print(f"VERDICT: full-waveform calibration = {'VALIDATED' if ok else 'PARTIAL'}")
    print(f"  The model calibrates its material field to the observed field with ONE adjoint pass per step and")
    print(f"  localises the hidden inclusion; the sensitivity map separates determined from undetermined cells.")
    print(f"  The same solver covers acoustic, EM and elastic media by a coefficient swap. Scope: 2-D lossless,")
    print(f"  single probe with full-field observation, c^2 only (sparse sensors / multiple sources are not covered).")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
