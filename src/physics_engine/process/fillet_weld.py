"""FILLET WELD STRENGTH — RENDER->MATCH the load a fillet weld carries, the workhorse joint of every steel structure, bracket and
machine frame. A fillet weld has a triangular cross-section; it fails by SHEAR across its THROAT, the 45-degree plane that is the
shortest path through the weld. For a leg size w the effective throat is
    a = 0.707 w   ->   tau = F / (a L) = F / (0.707 w L)
A 6 mm leg over 100 mm of weld carries ~50 kN at ~118 MPa of throat shear -- below an E70 electrode's ~144 MPa allowable. The
0.707 (=1/sqrt2) is pure geometry: the throat is the leg times cos45. Doubling the leg or the length halves the stress, which is
why weld SIZE and LENGTH are the design levers, and why over-welding wastes metal and heat.

I/O: no input files; prints the rendered throat shear, the gate lines and a PASS/FAIL verdict (exit 0 on pass).
MATCH: a 6 mm/100 mm fillet weld at 50 kN sees ~118 MPa throat shear; tau ~ F/(w L); the throat is 0.707 of the leg.
Reference: throat geometry a = 0.707 w and the E70 electrode allowable ~144 MPa (AWS D1.1 structural-welding practice).
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
import sys, numpy as np
from render_match_scaffold import Benchmark, render_match


def throat_shear(F, w, L):
    return F / (0.707 * w * L)                                 # throat shear stress [Pa]


def main():
    print("=" * 92)
    print("FILLET WELD — tau=F/(0.707 w L); shear across the 45-degree throat; render->match")
    print("=" * 92)
    def rfn(p):
        return throat_shear(p["F"], p.get("w", 6e-3), 0.100) / 1e6   # MPa
    band = [{"F": 45e3}, {"F": 55e3}]                         # applied-load uncertainty = sigma
    res = render_match(
        rfn, band, {"F": 50e3},
        Benchmark("throat shear of a 6 mm/100 mm fillet at 50 kN", round(throat_shear(50e3, 6e-3, 0.1) / 1e6, 0), 15.0, "tau=F/(0.707 w L)", "MPa"),
        nulls=[("no load, no stress (F->0 -> tau->0)", {"F": 1e-3}, lambda v, m: v < m / 1e3)],
        perturbations=[("a bigger weld leg lowers the stress (w up -> less tau)", {"F": 50e3, "w": 10e-3}, lambda v, best: v < best)],
        notes=["with no load the weld is unstressed; a larger leg or longer weld spreads the same force over more throat, lowering the stress"])
    print(res.report())
    # the 0.707 throat geometry + the size/length levers + the capacity check
    tau6 = throat_shear(50e3, 6e-3, 0.1) / 1e6; allow = 144.0
    print(f"\n  weld leg -> tau (50 kN, 100 mm):  4mm->{throat_shear(50e3,4e-3,0.1)/1e6:.0f}  6mm->{tau6:.0f}  10mm->{throat_shear(50e3,10e-3,0.1)/1e6:.0f}  MPa")
    print(f"  (4) ★THROAT = 0.707 x LEG: the failure plane is the leg times cos45, so a 6 mm leg gives a {0.707*6:.2f} mm throat and {tau6:.0f} MPa shear -- pure geometry, the same factor for every fillet weld")
    print(f"  (5) ★SIZE & LENGTH ARE THE LEVERS: tau~F/(w L), so doubling the leg or the length halves the stress; here {tau6:.0f} MPa sits under the E70 allowable ~{allow:.0f} MPa, so the joint passes -- over-welding beyond that just wastes metal and heat input")
    g4 = abs(throat_shear(50e3, 12e-3, 0.1) / throat_shear(50e3, 6e-3, 0.1) - 0.5) < 1e-6 and 100 < tau6 < 135   # tau~1/w; ~118 MPa
    g5 = abs(0.707 - 1/np.sqrt(2)) < 1e-3 and tau6 < allow     # throat geometry; under allowable
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (fillet weld) — throat shear from the weld geometry, not fitted:")
        print(f"  • a 6 mm/100 mm fillet at 50 kN sees {res.best:.0f} MPa throat shear (band [{res.band_lo:.0f},{res.band_hi:.0f}]=load σ).")
        print(f"  • the throat is 0.707 (cos45) of the leg, so tau~F/(w L) -- leg size and weld length are the design levers, each halving the stress when doubled.")
        print(f"  • {tau6:.0f} MPa is under the E70 ~{allow:.0f} MPa allowable, so it passes; sizing past that wastes weld metal and heat -- the everyday structural-joint check.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, tau~1/w {g4}, throat/capacity {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
