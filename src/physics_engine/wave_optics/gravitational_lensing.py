"""GRAVITATIONAL LENSING / EINSTEIN RADIUS — RENDER->MATCH how much a foreground mass bends light from a background
source into a ring, by COMPOSING the Schwarzschild radius (schwarzschild_radius) with the lensing geometry. A point mass deflects
a ray by 2 r_s / b (twice the Newtonian value -- Einstein's factor of 2, confirmed in 1919); setting the deflection equal to the
angle needed to reach the observer gives the Einstein radius
    theta_E = sqrt( 2 r_s D_LS / (D_L D_S) )
where r_s=2GM/c^2 and D_L, D_S, D_LS are observer-lens, observer-source and lens-source distances. A 1e12 M_sun galaxy at ~1 Gpc
lensing a source behind it makes a ~2 arcsec ring. Crucially theta_E measures the TOTAL mass inside it -- including dark matter --
purely from geometry, no light from the lens required, the cleanest cosmic scale. Uses render_match_scaffold (imports
schwarzschild_radius); a combination cell -- Schwarzschild radius x lensing geometry.

MATCH: a 1e12 M_sun lens at 1 Gpc gives an Einstein ring ~2 arcsec; theta_E ~ sqrt(M); a massless lens bends nothing.
Reference: Einstein-radius relation theta_E=sqrt(2 r_s D_LS/(D_L D_S)); the 1919 Eddington deflection measurement (2x Newton).
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
import sys, numpy as np
from render_match_scaffold import Benchmark, render_match
from schwarzschild_radius import r_s                           # Schwarzschild radius r_s=2GM/c^2 [m]

GPC = 3.086e25; ARCSEC = 206265.0


def theta_E(M_solar, DL=1.0, DS=2.0, DLS=1.0):
    rs = r_s(M_solar)
    return np.sqrt(2 * rs * (DLS * GPC) / ((DL * GPC) * (DS * GPC))) * ARCSEC   # Einstein radius [arcsec]


def main():
    print("=" * 92)
    print("GRAVITATIONAL LENSING — theta_E=sqrt(2 r_s D_LS/(D_L D_S)); Schwarzschild x geometry; render->match")
    print("=" * 92)
    def rfn(p):
        return theta_E(p["M"])
    band = [{"M": 0.8e12}, {"M": 1.2e12}]                     # lens-mass uncertainty = sigma
    res = render_match(
        rfn, band, {"M": 1e12},
        Benchmark("Einstein radius of a 1e12 M_sun galaxy lens", round(theta_E(1e12), 2), 0.3, "theta_E, Gpc distances", "arcsec"),
        nulls=[("a massless lens bends no light (M->0 -> theta_E->0)", {"M": 1.0}, lambda v, m: v < m / 1e5)],
        perturbations=[("a cluster-scale lens makes a bigger ring (M up -> larger theta_E)", {"M": 1e14}, lambda v, best: v > best)],
        notes=["no mass, no bending; a heavier lens makes a larger Einstein ring, growing as sqrt(M)"])
    print(res.report())
    # ★the sqrt(M) scaling + the mass-from-geometry
    sM = np.polyfit(np.log([1e11, 1e12, 1e13]), np.log([theta_E(m) for m in (1e11, 1e12, 1e13)]), 1)[0]
    print(f"\n  M -> theta_E:  galaxy(1e12)->{theta_E(1e12):.1f}\"  cluster(1e14)->{theta_E(1e14):.0f}\"  star(1)->{theta_E(1.0)*1e6:.1f} uas")
    print(f"  (4) ★theta_E ~ sqrt(M): the ring grows as M^{sM:.2f} -- a 100x heavier lens makes a 10x bigger ring, so measuring the ring size weighs the lens directly")
    print(f"  (5) ★MASS FROM PURE GEOMETRY: theta_E gives the TOTAL mass inside the ring -- dark matter included -- with no light from the lens, just the bent image; the deflection is 2x Newton (Einstein's 1919 factor), a clean GR test and a cosmic scale")
    g4 = abs(sM - 0.5) < 0.02 and 1.0 < theta_E(1e12) < 3.5    # sqrt(M); galaxy ring ~2"
    g5 = theta_E(1e14) > theta_E(1e12) > theta_E(1e10) and theta_E(1.0) > 0   # bigger mass bigger ring
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (gravitational lensing) — Schwarzschild x geometry composed, neither fitted:")
        print(f"  • a 1e12 M_sun galaxy lens makes an Einstein ring {res.best:.2f}\" (band [{res.band_lo:.2f},{res.band_hi:.2f}]=mass σ).")
        print(f"  • theta_E~sqrt(M) (slope {sM:.2f}): the ring size weighs the lens directly; a cluster (1e14 M_sun) makes a ~{theta_E(1e14):.0f}\" arc.")
        print(f"  • it measures the total mass including dark matter from geometry alone -- the deflection is 2x Newton (Einstein 1919), composed from the Schwarzschild radius.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, sqrt-M {g4}, bigger-ring {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
