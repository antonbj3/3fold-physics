"""AIRY DIFFRACTION LIMIT — RENDER->MATCH why no telescope or microscope can resolve detail finer than ~lambda/D: a
circular aperture of diameter D, however perfect, smears a point source into an AIRY DISK whose first dark ring sits at the angle
    sin(theta_1) = 1.22 lambda / D ,
and two points closer than that blur into one (the Rayleigh criterion). The 1.22 is NOT a fudge factor: it is the first zero of
the Bessel function J1 divided by pi (3.8317/pi), and it EMERGES here from the Fraunhofer diffraction pattern -- the squared FFT
of the circular aperture -- whose first radial null we LOCATE numerically. (The same J1 zero 3.8317 sets the 2nd mode of a circular
drum in circular_membrane.py: one Bessel zero, two physics -- optics and acoustics.) We compute the diffraction pattern, measure
the first-null factor, match 1.22, and show the limit: an infinite aperture has no diffraction limit; redder light resolves
coarser. Uses render_match_scaffold.

MATCH: the Airy first-null factor is 1.22=J1_zero/pi (emerges from the diffraction FFT); an infinite aperture -> no limit (angle->0); a longer wavelength resolves coarser.
Reference: Born & Wolf, Principles of Optics (Airy pattern, Rayleigh criterion); J1 first zero 3.8317.
"""
# --- sibling-package bootstrap: this repository splits the modules by domain, so put every
# --- package directory on sys.path when the file is run as a script.
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in sorted(_os.listdir(_PKG_ROOT)):
    _d = _os.path.join(_PKG_ROOT, _p)
    if _os.path.isdir(_d) and not _p.startswith('__') and _d not in _sys.path:
        _sys.path.insert(0, _d)
del _os, _sys, _p, _d
import sys
import numpy as np
from render_match_scaffold import Benchmark, render_match

LAMBDA0 = 550e-9                                          # green light [m]


def airy_factor(N=2400, ap_frac=0.08):
    """Fraunhofer FFT of a circular aperture; locate the first radial null; return the resolution factor (=1.22)."""
    x = np.linspace(-1, 1, N)
    X, Y = np.meshgrid(x, x)
    aperture = (np.sqrt(X ** 2 + Y ** 2) <= ap_frac).astype(float)
    I = np.abs(np.fft.fftshift(np.fft.fft2(aperture))) ** 2
    c = N // 2
    row = I[c, c:]                                        # radial cut from the center outward
    row = row / row[0]
    # FIRST local minimum = first dark ring (not the first below a fixed floor; the discrete FFT first null is shallow)
    i_zero = int(np.where(np.diff(row) > 0)[0][0])
    if 0 < i_zero < len(row) - 1:                        # parabolic sub-pixel refine of the minimum
        y0, y1, y2 = row[i_zero - 1], row[i_zero], row[i_zero + 1]
        denom = y0 - 2 * y1 + y2
        if denom != 0:
            i_zero = i_zero + 0.5 * (y0 - y2) / denom
    return float(i_zero * ap_frac)                        # factor = (first-null pixel) * aperture-fraction = J1_zero/pi


def resolution_angle(D, lam=LAMBDA0):
    return float(airy_factor() * lam / D)                # sin(theta_1) ~ 1.22 lambda/D


def main():
    print("=" * 96)
    print("AIRY DIFFRACTION LIMIT — first null 1.22=J1_zero/pi emerges from the diffraction FFT; render->match")
    print("=" * 96)
    D0 = 0.1                                              # 10 cm aperture
    def rfn(p):
        return resolution_angle(p.get("D", D0), lam=p.get("lam", LAMBDA0)) * 1e6   # microradians
    band = [{"D": 0.05}, {"D": 0.2}]                     # aperture-diameter uncertainty = sigma
    bench = 1.22 * LAMBDA0 / D0 * 1e6
    res = render_match(
        rfn, band, {"D": D0},
        Benchmark("diffraction-limited resolution angle", round(bench, 3), 0.4, "1.22 lambda/D (EXTERNAL)", "urad"),
        nulls=[("an infinite aperture has no diffraction limit -> the angle vanishes (D->inf -> theta->0)", {"D": 1e6}, lambda v, m: v < m / 1e4)],
        perturbations=[("a longer wavelength resolves COARSER -- redder light, bigger blur (lambda up -> larger angle)", {"lam": 1100e-9}, lambda v, best: v > best)],
        notes=["the resolution angle is 1.22 lambda/D; an infinite aperture has no diffraction limit, and a longer wavelength gives a coarser (larger) resolution angle"])
    print(res.report())
    # ★the emergent 1.22 + the J1-zero connection + the scaling
    f = airy_factor(); J1_zero = 3.831706; f_an = J1_zero / np.pi
    th = resolution_angle(D0) * 1e6
    print(f"\n  D -> resolution:  " + "  ".join(f"{d*100:.0f}cm:{resolution_angle(d)*1e6:.2f}urad" for d in (0.05, 0.1, 0.2)))
    print(f"  measured first-null factor={f:.4f}  vs J1_zero/pi=3.8317/pi={f_an:.4f}  (match {abs(f-f_an)/f_an*100:.1f}%)   [the textbook 1.22]")
    print(f"  (4) ★THE 1.22 EMERGES FROM DIFFRACTION: the squared FFT of the circular aperture (the Fraunhofer pattern) has its first dark ring at a factor {f:.3f}, matching J1_zero/pi={f_an:.3f}={f_an:.2f} to {abs(f-f_an)/f_an*100:.0f}% -- the 1.22 is the first zero of the Bessel J1 over pi, not a fudge; it falls straight out of the circular geometry's transform")
    print(f"  (5) ★ONE BESSEL ZERO, TWO PHYSICS: this same J1 zero 3.8317 sets the 2nd vibrational mode of a circular drum ([[circular_membrane]]) -- optics and acoustics share it because both solve the SAME Helmholtz problem on a disk. Practically: a 10cm aperture resolves ~{th:.0f} urad in green light; halve D and you double the blur, redshift and you blur more -- the hard floor under every telescope, microscope and camera lens")
    g4 = abs(f - f_an) / f_an < 0.03 and abs(resolution_angle(0.05) / resolution_angle(0.1) - 2.0) < 0.05   # 1.22 emerges; 1/D scaling
    g5 = resolution_angle(1e6) < resolution_angle(D0) / 1e4 and resolution_angle(D0, 1100e-9) > resolution_angle(D0)  # null; lambda-up coarser
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Airy diffraction limit) — the 1.22 resolution factor, derived from diffraction:")
        print(f"  • the resolution angle is {res.best:.3f} urad=1.22 lambda/D (band [{res.band_lo:.3f},{res.band_hi:.3f}] urad=aperture σ), the 1.22 emerging from the FFT to {abs(f-f_an)/f_an*100:.0f}%.")
        print(f"  • the 1.22 is J1_zero/pi=3.8317/pi -- the first Bessel zero, shared with the circular drum (optics<->acoustics).")
        print(f"  • infinite aperture -> no limit; redder light -> coarser; the diffraction floor under every telescope and microscope.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, 1.22/D-scaling {g4}, null/lambda-coarser {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
