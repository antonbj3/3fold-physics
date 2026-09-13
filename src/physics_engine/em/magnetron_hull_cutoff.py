"""MAGNETRON HULL CUTOFF — RENDER->MATCH the magnetic field that switches a crossed-field
(magnetron) tube from conducting to cut off: the principle behind every microwave-oven and radar magnetron, and the limit on
crossed-field amplifiers and Penning/magnetron ion gauges. In a planar diode an electron leaves the cathode and is pulled toward
the anode by the field E=V_a/d, but a transverse magnetic field B curls its path into a CYCLOID. Below a critical B the cycloid
still reaches the anode (current flows); above the Hull cutoff B_H the electron arcs back to the cathode before reaching the anode
and the diode is CUT OFF. We do NOT assert B_H: we integrate the electron equation of motion in the crossed E,B fields and find the
B at which the maximum excursion just equals the gap d -- which lands on the Hull value B_H=(1/d)sqrt(2 m V_a/e). The anode voltage
V_a is the physical σ (B_H ~ sqrt(V_a), the Hull cutoff parabola). render_match_scaffold.

MATCH: the Hull cutoff field, from integrating the electron cycloid in crossed E,B fields until its peak excursion equals the gap, equals B_H=(1/d)sqrt(2 m V_a/e); with no field the electron flies straight to the anode (no cutoff); a higher anode voltage needs a stronger field (B_H ~ sqrt(V_a)).
  python em/magnetron_hull_cutoff.py
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
from scipy.integrate import solve_ivp
from render_match_scaffold import Benchmark, render_match

ME, QE = 9.109e-31, 1.602e-19
VA0, D0 = 1000.0, 1e-3                    # anode voltage (physical σ); cathode-anode gap


def max_excursion(B, Va=VA0, d=D0):
    """integrate the electron in crossed E=V_a/d, B fields; return (peak x / d, reached_anode?)."""
    Ex = -Va / d
    def rhs(t, s):
        x, y, vx, vy = s
        return [vx, vy, (-QE / ME) * (Ex + vy * B), (-QE / ME) * (-vx * B)]
    def hit(t, s):
        return s[0] - d
    hit.terminal = True; hit.direction = 1
    Tc = 2 * np.pi * ME / (QE * B)
    sol = solve_ivp(rhs, [0, 3 * Tc], [0, 0, 0, 0], events=hit, rtol=1e-9, atol=1e-13, max_step=Tc / 2000)
    return sol.y[0].max() / d, (len(sol.t_events[0]) > 0)


def B_hull(Va=VA0, d=D0):
    """genuine cutoff: bisection on B for the trajectory whose peak excursion just reaches the anode."""
    lo, hi = 0.3 * (1 / d) * np.sqrt(2 * ME * Va / QE), 3.0 * (1 / d) * np.sqrt(2 * ME * Va / QE)
    for _ in range(34):
        B = 0.5 * (lo + hi)
        if max_excursion(B, Va, d)[1]:
            lo = B
        else:
            hi = B
    return 0.5 * (lo + hi)


def main():
    print("=" * 96)
    print("MAGNETRON HULL CUTOFF — cutoff field from the electron cycloid in crossed E,B; render->match")
    print("=" * 96)
    def rfn(p):
        if p.get("nofield"):
            return max_excursion(p.get("B", 1e-6), p.get("Va", VA0))[0]   # null branch: peak excursion / d
        return B_hull(Va=p.get("Va", VA0))
    band = [{"Va": 900.0}, {"Va": 1100.0}]    # anode-voltage σ: B_H ~ sqrt(V_a) (two-sided)
    res = render_match(
        rfn, band, {"Va": VA0},
        Benchmark("Hull cutoff field B_H", (1 / D0) * np.sqrt(2 * ME * VA0 / QE), 0.01 * (1 / D0) * np.sqrt(2 * ME * VA0 / QE), "(1/d) sqrt(2 m V_a/e) (Hull, EXTERNAL)", "T"),
        nulls=[("with essentially NO magnetic field (B->0) the electron is not curled at all -- it flies in a straight line straight across the gap and STRIKES the anode, its peak excursion reaching the full gap d (no cutoff exists: the diode conducts at any voltage)", {"nofield": True, "B": 1e-6}, lambda v, m: v >= 0.99)],
        perturbations=[("a HIGHER anode voltage pulls the electron across harder, so it takes a STRONGER field to curl it back before it reaches the anode -- the Hull cutoff rises as B_H ~ sqrt(V_a) (V_a up -> B_H up)", {"Va": 2000.0}, lambda v, best: v > 1.3 * best)],
        notes=["the cutoff field from the integrated cycloid matches (1/d) sqrt(2 m V_a/e); no field lets the electron reach the anode; B_H grows as sqrt(V_a)"])
    print(res.report())
    BH = B_hull(); Bf = (1 / D0) * np.sqrt(2 * ME * VA0 / QE)
    print(f"\n  ★CUTOFF FROM THE CYCLOID (integrated, not asserted): B_H = {BH:.5f} T vs (1/d) sqrt(2 m V_a/e) = {Bf:.5f} T ({abs(BH-Bf)/Bf*100:.2f}%) -- the equation of motion in crossed E,B is integrated and the field at which the peak excursion equals d IS the Hull cutoff; no cutoff formula was used to propagate the orbit")
    print(f"  ★THE CYCLOID & TURNING POINT (the mechanism): below B_H the electron's curved path still spans the gap (reaches anode); at B_H its peak x just grazes the anode; above, it arcs back to the cathode -- a single transverse B switches the tube from ON to OFF. peak excursion x_max=2 m E/(e B^2)=d at cutoff")
    print(f"  ★B_H ~ sqrt(V_a) (the Hull parabola, the σ): B_H = " + ", ".join(f"Va={v}:{(1/D0)*np.sqrt(2*ME*v/QE):.4f}T" for v in (500, 1000, 2000)) + " -- the cutoff boundary in the (V_a,B) plane is the parabola V_a=(e/2m) B^2 d^2; magnetrons are biased just inside it so the rotating space-charge spokes can bunch and radiate")
    print(f"  (4) ★WHY IT MATTERS: the Hull cutoff sets the operating point of every magnetron (microwave ovens, marine/air radar), crossed-field amplifier, magnetron sputter source and Penning/magnetron vacuum gauge; the crossed-field electron dynamics are the kernel a vacuum-electronics or plasma-device digital twin integrates")
    g4 = abs(BH - Bf) / Bf < 0.02 and abs(B_hull(2000.0) / B_hull(500.0) - 2.0) < 0.1 and max_excursion(0.5 * Bf)[1]  # cutoff matches; sqrt(Va) (2000/500=4 -> 2x); below-cutoff reaches
    g5 = max_excursion(1e-6)[0] >= 0.99 and B_hull(2000.0) > 1.3 * BH                                                # no-field null; higher Va perturbation
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/magnetron_hull_cutoff.json", "w") as fh:
        json.dump({"module": "magnetron_hull_cutoff", "provenance": "self-contained crossed-field electron trajectory integration, no external data",
                   "B_hull_trajectory": float(BH), "B_hull_formula": float(Bf),
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "nofield_reaches_anode": float(max_excursion(1e-6)[0]), "B_hull_2000V_over_500V": float(B_hull(2000.0) / B_hull(500.0)),
                   "cross_checks": {"cutoff_sqrtVa_belowreaches": bool(g4), "null_and_higherVa": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (magnetron Hull cutoff) — a transverse field switching a diode off:")
        print(f"  • cutoff B_H={BH:.5f} T (integrated cycloid) matches (1/d) sqrt(2 m V_a/e)={Bf:.5f} T (band [{res.band_lo:.4f},{res.band_hi:.4f}]=voltage σ).")
        print(f"  • no field lets the electron reach the anode; B_H~sqrt(V_a) (the Hull parabola).")
        print(f"  • ★a distinct vacuum-electronics/crossed-field primitive (charged-particle dynamics) for magnetron twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, cutoff/sqrtVa {g4}, null/Va {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
