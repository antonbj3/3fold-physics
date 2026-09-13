"""Wave-to-ray handoff, measured: when is ray tracing the correct solver for a macro lens, and when is a full-wave
solve required? The departure is justified by measurement, not assertion.

Two measured facts:
  (A) HANDOFF: the exact scalar-wave solution (angular-spectrum propagation by FFT) converges to geometric optics as
      lambda/D -> 0 - the focal spot forms at the geometric focus f, and its width (the diffraction limit ~ lambda*f/D)
      shrinks proportionally to lambda/D relative to the aperture. Ray tracing is therefore the lambda/D -> 0 limit of
      the wave solution.
  (B) GRID INFEASIBILITY: a direct FDTD of the same lens needs at least N_ppw*(D/lambda) cells per dimension; for a
      macro lens (D/lambda ~ 1e4-1e5) that is ~1e8-1e10 cells in 2D (1e12+ in 3D) - a closed-form Nyquist
      points-per-wavelength estimate, not a simulation - beyond feasible memory, so ray/paraxial is required.

Conclusion: angular-spectrum/FDTD where features are of order lambda (diffraction matters); ray/paraxial where
D >> lambda (geometric limit). Wave optics is the ground truth; ray optics is its measured limit.

CHECKS: (1) the wave solution focuses at the geometric focus z = f (recovered from the wavefront, not assumed);
(2) focal-spot width equals the diffraction limit lambda*f/D (Abbe/Rayleigh), measured vs analytic; (3) as
lambda/D -> 0, spot/D is proportional to lambda/D (log-log slope ~1); (4) the FDTD cell count for a macro lens exceeds
feasible memory.

INPUT: none. OUTPUT: printed gate lines + `artifacts/optics_departure.png`.
"""
import os
import sys
import numpy as np

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def focus_wave(D_over_lam, f_over_D=4.0, N=8192, pad=6.0):
    """1D angular-spectrum: plane wave through an aperture D with a thin-lens phase, propagated to the focal region.
    Returns (z_focus_measured/f, spot_FWHM, lam, f, D). All lengths in units where λ=1."""
    lam = 1.0; k = 2 * np.pi / lam
    D = D_over_lam * lam; f = f_over_D * D
    L = pad * D                                                  # transverse window (aperture + padding)
    x = (np.arange(N) - N // 2) * (L / N); dx = L / N
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ap = (np.abs(x) <= D / 2).astype(float)                     # hard aperture
    U0 = ap * np.exp(-1j * k * x**2 / (2 * f))                  # thin-lens converging phase
    A0 = np.fft.fft(U0)
    kz = np.sqrt(np.maximum(k**2 - kx**2, 0.0)) + 0j            # propagating angular spectrum (evanescent zeroed by max)
    # scan z near f to locate the peak-intensity (focus) plane
    zs = np.linspace(0.3 * f, 1.4 * f, 111); peak = []          # ★audit-fix: wider window (was 0.6f) — the low-Fresnel focus
    for z in zs:
        U = np.fft.ifft(A0 * np.exp(1j * kz * z))
        peak.append(np.max(np.abs(U)**2))
    zf = zs[int(np.argmax(peak))]
    # field at the measured focus → FWHM of |U|^2
    Uf = np.abs(np.fft.ifft(A0 * np.exp(1j * kz * zf)))**2
    Uf /= Uf.max()
    above = np.where(Uf >= 0.5)[0]
    fwhm = (above[-1] - above[0]) * dx if len(above) > 1 else dx
    return zf / f, fwhm, lam, f, D


def fdtd_cells(D_over_lam, f_over_D=4.0, ppw=15, dim=2):
    """cells a direct FDTD would need to resolve a lens of aperture D and length ~f at ppw points per wavelength."""
    span = (f_over_D + 1) * D_over_lam                          # ~ (f + D) in units of λ
    n_per_dim = ppw * span
    return n_per_dim ** dim


def main():
    print("=" * 84)
    print("OPTICS DEPARTURE — MEASURE the wave↔ray handoff (λ/D→0) + FDTD grid infeasibility")
    print("=" * 84)
    print(f"\n  exact scalar wave = angular-spectrum (FFT); ray/geometric = the λ/D→0 limit. units: λ=1, f=4·D")

    # (A) handoff: sweep D/λ, measure focus location + spot width vs the diffraction limit λf/D
    print(f"\n  (A) wave→ray handoff (sweep D/λ):")
    DoL = np.array([8, 16, 32, 64, 128]); zfr_list = []; spot_rel = []; dl_ratio = []
    for d in DoL:
        zfr, fwhm, lam, f, D = focus_wave(d)
        diff_limit = lam * f / D                                 # analytic diffraction-limited FWHM scale (~λf/D)
        zfr_list.append(zfr); spot_rel.append(fwhm / D); dl_ratio.append(fwhm / diff_limit)
        print(f"    D/λ={d:>4}: focus z/f={zfr:.3f} (geom=1) | spot FWHM/D={fwhm/D:.4f} | FWHM/(λf/D)={fwhm/diff_limit:.2f}")
    zfr_list = np.array(zfr_list); spot_rel = np.array(spot_rel); dl_ratio = np.array(dl_ratio)

    # the focus → geometric f as λ/D→0 (the diffractive FOCAL SHIFT vanishes at high Fresnel number — itself part of the handoff)
    ok1 = (zfr_list[-1] > 0.95) and np.all(np.diff(zfr_list) > 0)
    ok2 = (0.6 < np.median(dl_ratio) < 1.8)                     # spot width == the diffraction limit (order-unity prefactor)
    slope = float(np.polyfit(np.log(DoL), np.log(spot_rel), 1)[0])
    ok3 = -1.3 < slope < -0.7                                    # spot/D ∝ (λ/D)^1 → 0 : the geometric limit
    print(f"\n  (1) focus → GEOMETRIC focus f as λ/D→0: z/f {zfr_list[0]:.2f}→{zfr_list[-1]:.2f} (diffractive focal-shift vanishes)  {'✓' if ok1 else 'FAIL'}")
    print(f"  (2) focal spot == diffraction limit λf/D: FWHM/(λf/D)={np.median(dl_ratio):.2f} (order-unity)  {'✓' if ok2 else 'FAIL'}")
    print(f"  (3) λ/D→0 geometric limit: spot/D ∝ (λ/D)^{-slope:.2f} → 0  (ray-tracing IS the wave limit)  {'✓' if ok3 else 'FAIL'}")

    # (B) FDTD grid infeasibility at the macro-lens scale (measured cell counts)
    print(f"\n  (4) FDTD grid cost vs feature/λ (ppw=15):")
    feasible_2d = 1e9                                            # ~ a few ×10⁹ cells is a generous single-node 2D ceiling
    cross = None
    for dol in (1e2, 1e3, 1e4, 1e5):
        c2 = fdtd_cells(dol, dim=2); c3 = fdtd_cells(dol, dim=3)
        print(f"      D/λ={dol:.0e}: FDTD ~{c2:.1e} cells (2D), ~{c3:.1e} (3D)")
        if cross is None and c2 > feasible_2d:
            cross = dol
    ok4 = cross is not None and cross <= 1e4
    print(f"      ⇒ 2D FDTD crosses ~{feasible_2d:.0e}-cell feasibility at D/λ≈{cross:.0e}; a macro lens (D/λ~10⁴–10⁵) is")
    print(f"        infeasible by FDTD → ray/paraxial REQUIRED, justified by the measured handoff (A)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
        ax[0].loglog(DoL, spot_rel, "bo-", label="wave spot FWHM/D"); ax[0].loglog(DoL, spot_rel[0]*(DoL/DoL[0])**-1.0, "r--", lw=0.8, label="∝ λ/D (geometric)")
        ax[0].set_xlabel("D/λ"); ax[0].set_ylabel("spot/D"); ax[0].set_title("wave→ray handoff: spot→0 as λ/D→0", fontsize=8); ax[0].legend(fontsize=7)
        dd = np.logspace(1, 5, 40); ax[1].loglog(dd, [fdtd_cells(x, dim=2) for x in dd], "b-", label="FDTD cells 2D")
        ax[1].axhline(feasible_2d, color="k", ls=":", lw=0.8, label="feasible ~1e9"); ax[1].set_xlabel("D/λ"); ax[1].set_ylabel("cells")
        ax[1].set_title("FDTD grid infeasibility at macro scale", fontsize=8); ax[1].legend(fontsize=7)
        fig.tight_layout(); os.makedirs(_ART, exist_ok=True); fig.savefig(os.path.join(_ART, "optics_departure.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("OPTICS DEPARTURE — wave↔ray handoff MEASURED (FFT wave solution) + FDTD infeasibility ESTIMATED (Nyquist scaling):")
        print(f"  • the EXACT scalar-wave solution focuses at the geometric focus and its spot == the diffraction limit λf/D,")
        print(f"    shrinking ∝(λ/D)^{-slope:.1f} → ray-tracing IS the measured λ/D→0 limit of wave optics (a smooth handoff).")
        print(f"  • direct FDTD crosses single-node feasibility at D/λ≈{cross:.0e}; a macro lens is infeasible → ray/paraxial REQUIRED.")
        print(f"  ⇒ completes the honest multi-method reframe: BOTH major departures now MEASURED (compressible Mach-ceiling +")
        print(f"    optics wave/ray handoff). Full-wave where features are O(λ); ray where D≫λ. {'Render → artifacts/optics_departure.png' if rend else ''}")
    else:
        print(f"  (1)focus {ok1} (2)diff-limit {ok2} (3)geom-limit {ok3} (slope {slope:.2f}) (4)FDTD-infeasible {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
