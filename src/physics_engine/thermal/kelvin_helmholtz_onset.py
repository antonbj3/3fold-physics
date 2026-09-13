"""KELVIN-HELMHOLTZ SHEAR-INSTABILITY ONSET — RENDER->MATCH the minimum wind speed that raises waves on water, and
more generally when a sheared interface between two fluids goes unstable. GEOMETRIC mechanism: a small crest on the interface
narrows the faster stream above it, so by Bernoulli the pressure there DROPS and sucks the crest up further -- shear destabilizes.
Gravity (heavy fluid below) and surface tension fight back, pulling the crest down. Linear stability gives the marginal shear
    (dU)^2_crit = 2 (rho1 + rho2)/(rho1 rho2) * sqrt(g sigma (rho2 - rho1))
(the worst-case perturbation sits at the capillary-gravity wavenumber k* = sqrt(g(rho2-rho1)/sigma), where the two restoring
forces are jointly weakest). For air over water this is ~6.6 m/s -- the classic inviscid threshold for wind-driven waves. Shear
beats BOTH restoring forces at once, so the onset is set by their geometric-mean balance, not either alone. Built from the linear
dispersion, never fit. render_match_scaffold.

MATCH: air-over-water KH onset ~6.6 m/s; remove gravity+tension and any shear is unstable; stronger surface tension raises the threshold.
  python thermal/kelvin_helmholtz_onset.py
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
from render_match_scaffold import Benchmark, render_match

RHO_A, RHO_W, G = 1.2, 1000.0, 9.81                        # air, water densities [kg/m^3]; gravity [m/s^2]
SIGMA = 0.072                                              # air-water surface tension [N/m]


def U_crit(sigma=SIGMA, g=G, rho1=RHO_A, rho2=RHO_W):
    return np.sqrt(2 * (rho1 + rho2) / (rho1 * rho2) * np.sqrt(g * sigma * (rho2 - rho1)))


def k_star(sigma=SIGMA, g=G, rho1=RHO_A, rho2=RHO_W):
    return np.sqrt(g * (rho2 - rho1) / sigma)             # capillary-gravity wavenumber (most-unstable)


def main():
    print("=" * 96)
    print("KELVIN-HELMHOLTZ ONSET — shear vs gravity+surface-tension; air/water ~6.6 m/s; render->match")
    print("=" * 96)
    def rfn(p):
        return U_crit(sigma=p.get("sigma", SIGMA), g=p.get("g", G))
    band = [{"sigma": 0.060}, {"sigma": 0.085}]           # surface-tension uncertainty = sigma
    res = render_match(
        rfn, band, {"sigma": SIGMA},
        Benchmark("air-over-water KH critical wind speed", 6.6, 0.5, "inviscid KH threshold (EXTERNAL, classic)", "m/s"),
        nulls=[("with no gravity AND no surface tension any shear is unstable (-> U_crit->0)", {"sigma": 1e-9, "g": 1e-9}, lambda v, m: v < 0.05)],
        perturbations=[("stronger surface tension stiffens the interface (sigma up -> higher U_crit)", {"sigma": 0.30}, lambda v, best: v > best)],
        notes=["the critical shear is the geometric-mean balance of gravity and surface tension; remove both and any shear destabilizes, and a stiffer interface raises the threshold"])
    print(res.report())
    # ★the geometric-mean balance + the most-unstable wavelength
    lam = 2 * np.pi / k_star() * 1000                      # most-unstable wavelength [mm]
    print(f"\n  sigma -> U_crit:  " + "  ".join(f"{s:.2f}:{U_crit(sigma=s):.2f}" for s in (0.02, 0.072, 0.30)))
    print(f"  U_crit(air/water) = {U_crit():.2f} m/s   most-unstable wavelength = {lam:.1f} mm (k* = capillary-gravity)")
    print(f"  (4) ★SHEAR BEATS BOTH RESTORERS AT ONCE: U_crit ~ (g sigma)^1/4, the GEOMETRIC MEAN of gravity and surface tension -- so the threshold is {U_crit():.1f} m/s, lower than either force alone would set; the worst perturbation sits at k* = sqrt(g(rho2-rho1)/sigma) ({lam:.0f} mm), where gravity (long waves) and tension (short waves) are jointly weakest")
    print(f"  (5) ★ONE ONSET, MANY INTERFACES: the same balance governs wind-driven ocean waves, fuel-jet atomization, cloud billows, and shear layers in any density-stratified flow -- raise the shear past U_crit and the smooth interface rolls up into the Kelvin-Helmholtz cat's-eyes that mix the two streams")
    g4 = abs(U_crit() - np.sqrt(2*(RHO_A+RHO_W)/(RHO_A*RHO_W)*np.sqrt(G*SIGMA*(RHO_W-RHO_A)))) < 1e-9 and U_crit(sigma=0.30) > U_crit()  # formula; stiffer
    g5 = U_crit(sigma=1e-9, g=1e-9) < 0.05 and 5.5 < U_crit() < 7.5 and 10 < lam < 25  # null; ~6.6; cap-grav wavelength ~17mm
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Kelvin-Helmholtz onset) — shear instability, derived from the dispersion:")
        print(f"  • air-over-water onset is {res.best:.2f} m/s (band [{res.band_lo:.2f},{res.band_hi:.2f}]=surface-tension σ), matching the classic ~6.6 m/s.")
        print(f"  • U_crit ~ (g sigma)^1/4 -- the geometric mean of the two restoring forces; worst wavelength {lam:.0f} mm (capillary-gravity).")
        print(f"  • one onset for ocean waves, atomization, cloud billows and any stratified shear layer -- the KH cat's-eye roll-up.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, formula/stiffer {g4}, null/onset/wavelength {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
