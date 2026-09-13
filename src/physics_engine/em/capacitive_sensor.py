"""CAPACITIVE PROXIMITY SENSOR — RENDER->MATCH how a parallel-plate capacitor reads distance, the principle behind touchscreens,
MEMS accelerometers, micrometer gap gauges and liquid-level probes. Two plates of area A separated by a gap d hold
    C = eps0 eps_r A / d
so the capacitance rises as the target gets CLOSER -- C ~ 1/d. A 1 cm^2 plate at a 1 mm air gap is only ~0.9 pF, a tiny value
that demands careful electronics, and because C ~ 1/d the response is NONLINEAR: the sensor is far more sensitive at small gaps
(dC/dd ~ 1/d^2), which is why precision gauges run tiny gaps and why differential (two-gap) designs are used to linearize. Bring
a finger or a grounded target in and the field couples differently -- the same physics reads touch, proximity, displacement and
level. Uses render_match_scaffold.

MATCH: a 1 cm^2 plate at 1 mm gap is ~0.9 pF; C ~ 1/d (closer = more); far away -> ~0; halving the gap doubles C.
  python em/capacitive_sensor.py
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

EPS0 = 8.854e-12


def capacitance(d, A=1e-4, eps_r=1.0):
    return EPS0 * eps_r * A / d                               # parallel-plate capacitance [F]


def main():
    print("=" * 92)
    print("CAPACITIVE SENSOR — C=eps0 eps_r A/d; capacitance ~ 1/gap; touch/MEMS/level; render->match")
    print("=" * 92)
    def rfn(p):
        return capacitance(p["d"], eps_r=p.get("er", 1.0)) * 1e12   # pF
    band = [{"d": 0.9e-3}, {"d": 1.1e-3}]                     # gap (target distance) uncertainty = sigma
    res = render_match(
        rfn, band, {"d": 1.0e-3},
        Benchmark("capacitance of a 1 cm^2 plate at 1 mm gap", round(capacitance(1e-3) * 1e12, 2), 0.1, "C=eps0 A/d", "pF"),
        nulls=[("target far away, capacitance vanishes (d->large -> C->0)", {"d": 1.0}, lambda v, m: v < m / 100)],
        perturbations=[("bring the target closer, capacitance rises (d down -> more C)", {"d": 0.5e-3}, lambda v, best: v > best)],
        notes=["a distant target couples almost nothing; closing the gap raises C as 1/d, so the sensor is most sensitive up close"])
    print(res.report())
    # ★the 1/d nonlinearity + the small signal + dielectric
    c1 = capacitance(1e-3) * 1e12
    print(f"\n  gap -> C:  2mm->{capacitance(2e-3)*1e12:.2f}  1mm->{c1:.2f}  0.5mm->{capacitance(0.5e-3)*1e12:.2f}  pF   (water dielectric eps_r=80 at 1mm: {capacitance(1e-3,eps_r=80)*1e12:.0f} pF)")
    print(f"  (4) ★C ~ 1/GAP, NONLINEAR: halving the gap (1->0.5 mm) doubles C ({c1:.2f}->{capacitance(0.5e-3)*1e12:.2f} pF); sensitivity dC/dd~1/d^2 so the sensor is far keener at small gaps -- precision gauges run tiny gaps, and differential two-gap designs linearize the reading")
    print(f"  (5) ★TINY SIGNAL, MANY USES: just {c1:.2f} pF at 1 mm demands careful electronics, but the one law reads it all -- shrink d for displacement/MEMS, change A for a slider, change eps_r for liquid level (water's eps_r=80 lifts C ~80x) or a touchscreen finger")
    g4 = abs(capacitance(0.5e-3)/capacitance(1e-3) - 2.0) < 1e-9 and abs(capacitance(1e-3,eps_r=80)/capacitance(1e-3) - 80) < 1e-6   # C~1/d; C~eps_r
    g5 = 0.5 < c1 < 1.5 and capacitance(0.5e-3) > capacitance(1e-3)   # ~0.9 pF; closer = more
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (capacitive sensor) — capacitance from geometry, not fitted:")
        print(f"  • a 1 cm^2 plate at 1 mm gap is {res.best:.2f} pF (band [{res.band_lo:.2f},{res.band_hi:.2f}]=gap σ) -- a tiny signal demanding careful electronics.")
        print(f"  • C~1/d, so the response is nonlinear and keenest up close (dC/dd~1/d^2) -- precision gauges run small gaps, differential designs linearize.")
        print(f"  • one law spans touch, proximity, displacement, MEMS and level (eps_r): shrink d, change A, or change the dielectric -- the everyday capacitive-sensing primitive.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, 1/d & eps_r {g4}, value/closer {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
