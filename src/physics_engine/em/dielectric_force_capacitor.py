"""DIELECTRIC FORCE IN A CAPACITOR — MAXWELL STRESS / EM-ELASTIC COUPLING — RENDER->MATCH the force that SUCKS a slab
of dielectric into a charged capacitor -- the electric field reaching out of the gap (the fringing field) grabs the polarized
slab and pulls it in. It is the canonical electromechanical (EM<->elastic) coupling: a field exerts a mechanical force on matter.
The energy method gives it cleanly. With the slab inserted a depth x, the capacitance is C(x)=eps0 w [eps_r x + (L-x)]/d, and the
force follows from the field energy,
    F = (eps_r - 1) eps0 w V^2 / (2 d)        (pulling the slab IN, independent of x while it is partly inserted).
We do NOT assume it: we build C(x), differentiate the energy NUMERICALLY, and -- the key cross-check -- we compute it BOTH at
constant voltage (battery attached, F=+dU/dx) and at constant charge (isolated, F=-dU/dx). The two give the SAME force despite
opposite-sign energy bookkeeping (the battery silently does work) -- a thermodynamic-consistency check that no single calculation
can fake. render_match_scaffold.

MATCH: the force pulling a dielectric slab into a capacitor, from numerically differentiating the field energy, equals (eps_r-1) eps0 w V^2/(2d) and is IDENTICAL at constant voltage and constant charge; no dielectric (eps_r=1) gives no force; it scales as V^2.
  python em/dielectric_force_capacitor.py
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

EPS0, W, D, L = 1.0, 1.0, 0.1, 1.0                        # plate width W, gap D, length L (e=1 units)
X0, EPSR0, V0 = 0.5, 4.0, 2.0                             # slab partly inserted; dielectric constant; voltage


def C(x, epsr):
    return EPS0 * W * (epsr * x + (L - x)) / D            # capacitance with slab inserted depth x


def force_constV(x=X0, epsr=EPSR0, V=V0, h=1e-6):
    """force from the field energy at CONSTANT VOLTAGE (battery): F=+dU/dx, U=C V^2/2 (the battery does work)."""
    return float((V ** 2 / 2) * (C(x + h, epsr) - C(x - h, epsr)) / (2 * h))


def force_constQ(x=X0, epsr=EPSR0, V=V0, h=1e-6):
    """force at CONSTANT CHARGE (isolated): F=-dU/dx, U=Q^2/(2C); Q set so V matches at x (same physical state)."""
    Q = V * C(x, epsr)
    return float(-(Q ** 2 / 2) * (1 / C(x + h, epsr) - 1 / C(x - h, epsr)) / (2 * h))


def force_analytic(epsr=EPSR0, V=V0):
    return float((epsr - 1) * EPS0 * W * V ** 2 / (2 * D))


def main():
    print("=" * 96)
    print("DIELECTRIC FORCE — slab pulled into a capacitor; energy method, const-V == const-Q; render->match")
    print("=" * 96)
    def rfn(p):
        return force_constV(epsr=p.get("epsr", EPSR0), V=p.get("V", V0))
    band = [{"epsr": 0.9 * EPSR0}, {"epsr": 1.1 * EPSR0}]  # dielectric-constant σ (±10%): force ~ (eps_r - 1)
    res = render_match(
        rfn, band, {"epsr": EPSR0},
        Benchmark("dielectric pull-in force", force_analytic(), 0.04, "(eps_r-1) eps0 w V^2/(2d) (EXTERNAL)", "N"),
        nulls=[("with NO dielectric (eps_r=1, vacuum gap) there is nothing to polarize -- the fringing field grabs nothing, no force (eps_r->1 -> F->0)", {"epsr": 1.0}, lambda v, m: v < m / 10)],
        perturbations=[("a higher voltage stores more field energy and pulls harder -- the force scales as V^2 (V up -> F up steeply)", {"V": 2 * V0}, lambda v, best: v > best + 2.0)],
        notes=["the numerically-differentiated field energy gives (eps_r-1) eps0 w V^2/(2d); identical at constant V and constant Q; no dielectric gives no force"])
    print(res.report())
    # ★the const-V==const-Q consistency + the V^2/(eps-1) scaling + the null + the Maxwell-stress origin
    fv, fq, fa = force_constV(), force_constQ(), force_analytic()
    print(f"\n  pull-in force = {fv:.4f} (const-V) = {fq:.4f} (const-Q) vs analytic {fa:.4f}")
    print(f"  ★CONST-V == CONST-Q (thermodynamic consistency, the cross-method): the SAME force {fv:.2f} comes out at constant voltage (F=+dU/dx, battery DOES work) and constant charge (F=-dU/dx, isolated) -- opposite energy bookkeeping, identical mechanics. A force calc that only worked one way would be wrong")
    print(f"  ★PULLED IN, NOT OUT (the fringing field): the slab is sucked INTO the gap (F>0) -- the field is lower-energy with more dielectric, so the system pulls matter in to fill it. The force acts at the slab EDGE via the fringing field (the Maxwell stress), even though the uniform-field region exerts none")
    print(f"  ★V^2 and (eps_r-1): F = " + ", ".join(f"V={v}:{force_constV(V=v):.1f}" for v in (1, 2, 4)) + " (quadratic in V); and F = " + ", ".join(f"eps={e}:{force_constV(epsr=e):.1f}" for e in (1, 4, 8)) + " (linear in eps_r-1) -- a vacuum gap (eps_r=1) feels nothing; the polarizability is the whole effect")
    print(f"  (4) ★FORCE FROM THE ENERGY (not assumed): differentiating C(x) numerically and forming dU/dx gives {fv:.4f}, matching (eps_r-1)eps0 w V^2/(2d) to {abs(fv-fa)/fa*100:.3f}% -- the electromechanical force is purely the field's energy wanting to grow; no separate 'electrostriction coefficient' is needed for the rigid-slab force")
    print(f"  (5) ★WHY IT MATTERS: this Maxwell-stress force is the actuator principle of MEMS/NEMS (comb drives, variable capacitors), dielectric elastomer actuators and electroadhesion, liquid-dielectric electrowetting and lab-on-chip pumping, and is the electric twin of the magnetic pressure that shapes plasmas and rail-guns -- a field doing mechanical work on matter")
    g4 = abs(fv - fa) / fa < 0.01 and abs(fv - fq) / fv < 1e-3 and force_constV(epsr=1.0) < 1e-6  # numerical=analytic; const-V==const-Q; null
    g5 = abs(force_constV(V=2 * V0) / fv - 4.0) < 0.01 and force_constV(epsr=8.0) > force_constV(epsr=4.0) > force_constV(epsr=2.0)  # V^2 scaling; eps monotone
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (dielectric force) — the capacitor that swallows a slab:")
        print(f"  • pull-in force {res.best:.4f} matches (eps_r-1)eps0 w V^2/(2d) (band [{res.band_lo:.4f},{res.band_hi:.4f}]=dielectric-constant σ).")
        print(f"  • identical at constant V and constant Q; pulled IN by the fringing field; V^2 and (eps_r-1) scaling; vacuum feels nothing.")
        print(f"  • ★a field doing mechanical work on matter -- the Maxwell-stress actuator principle, derived from energy alone.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, numerical/constV-constQ/null {g4}, V^2/eps {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
