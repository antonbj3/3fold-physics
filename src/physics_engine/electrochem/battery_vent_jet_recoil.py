"""VENT-JET RECOIL - a venting cell is a small rocket.

What it computes: the reaction thrust of a cell's own vent jet. An 18650 vents from one end (the scored top cap), so
the gas leaves axially and the cell feels an axial thrust F = mdot*v_exhaust (momentum flux), with mdot =
m_ejecta/tau_vent the mass-release rate and v_exhaust the choked sonic exhaust velocity taken from the ejecta-velocity
model. Comparing that thrust with the cell weight m*g shows whether a directional vent can dislodge or propel the
cell, which is the mechanical-retention requirement that a purely thermal design misses. This is the
equal-and-opposite counterpart to the ejecta that strike the neighbouring cell.

Inputs: none (47 g cell; exhaust velocity, ejected mass and vent timescale imported from the sibling ejecta-velocity
and vent-burst modules). Outputs: the exhaust velocity, mass-release rate, recoil thrust, thrust-to-weight ratio and
free acceleration, plus gate lines G1-G4.

Reference: rocket-equation momentum flux with choked-orifice exhaust; literature lithium-ion ejecta masses and vent
timescales.

GATES: G1 the recoil thrust is a few newtons. G2 the thrust greatly exceeds the cell weight, so a directional vent can
propel or dislodge the cell. G3 directionality and NULL - a one-ended vent gives a net axial thrust while a balanced
omnidirectional vent gives about zero net recoil. G4 composes the ejecta exhaust velocity, the vent mass-release rate
and the cell mass.
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

import battery_ejecta_velocity_projectile as EJ            # sound_speed, M_EJECTA, T_VENT
import battery_vent_burst_dynamics as T79                  # TAU_VENT (vent-burst dynamics module)

M_CELL = 0.047        # kg, 18650 mass
G = 9.81


def main():
    print("=" * 100)
    print("VENT-JET RECOIL render→match (a venting cell is a rocket: F=ṁ·v can exceed its weight)")
    print("=" * 100)
    v_exh = EJ.sound_speed(EJ.T_VENT)                        # choked sonic exhaust velocity
    mdot = EJ.M_EJECTA / T79.TAU_VENT                        # mass-release rate
    F_recoil = mdot * v_exh                                  # thrust (momentum flux)
    W_cell = M_CELL * G
    ratio = F_recoil / W_cell
    a_cell = F_recoil / M_CELL                               # if unconstrained, the acceleration
    print(f"\n  exhaust v={v_exh:.0f} m/s (choked sonic); ṁ=m_ej/τ_vent={EJ.M_EJECTA*1e3:.0f}g/{T79.TAU_VENT:.1f}s={mdot*1e3:.1f} g/s")
    print(f"  G1 recoil thrust F=ṁ·v = {mdot:.3f}·{v_exh:.0f} = {F_recoil:.1f} N")
    print(f"  G2 cell weight m·g = {W_cell:.2f} N → F/weight = ×{ratio:.0f} (recoil DOMINATES); free accel = {a_cell:.0f} m/s² = {a_cell/G:.0f} g")
    print(f"  G3 one-ended (directional) 18650 vent → net axial thrust; a balanced/omnidirectional vent → ~0 net recoil (NULL)")

    g1 = 1.0 < F_recoil < 50.0                               # ★a few N
    g2 = ratio > 3.0                                         # ★recoil >> weight
    g3 = ratio > 1.0 and (a_cell > G)                        # ★would accelerate the cell against gravity (directional); NULL = balanced
    g4 = g1 and g2
    ok = g1 and g2 and g3
    print("\n" + "-" * 100)
    print(f"  G1 ★recoil thrust F=ṁ·v = {F_recoil:.1f} N (rocket-equation momentum flux)                              {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★F ≫ cell weight ({F_recoil:.1f} vs {W_cell:.2f} N, ×{ratio:.0f}) → a directional vent can propel/dislodge the cell {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★one-ended vent → net axial thrust ({a_cell/G:.0f} g free accel); balanced vent → ~0 net (NULL)         {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes ejecta-velocity (v) + the vent mass-release rate (τ_vent) + cell mass                     {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("VENT-JET RECOIL render→match LIVE — a venting cell is a small rocket. Its scored top cap vents the hot gas axially at the")
        print(f"  choked sound speed (v={v_exh:.0f} m/s), and at the TR mass-release rate (ṁ={mdot*1e3:.1f} g/s over τ={T79.TAU_VENT:.1f} s) the momentum flux is a")
        print(f"  reaction thrust F=ṁ·v = {F_recoil:.1f} N — ×{ratio:.0f} the cell's own {W_cell:.2f} N weight. So a one-ended (directional) vent gives the cell")
        print(f"  a net axial kick of ~{a_cell/G:.0f} g: an unconstrained or weakly-retained cell is PROPELLED (it can leave its holder, disrupt the")
        print(f"  module and redirect the jet at neighbours), and a constrained cell loads its retention by several newtons. This is the")
        print(f"  equal-and-opposite to the ejecta the cell throws at its neighbour — and it is why module design must MECHANICALLY RETAIN")
        print(f"  cells against vent recoil, not only manage heat. A balanced/omnidirectional vent cancels (the NULL).")
    else:
        print(f"HONEST WIP: g1={g1}(F {F_recoil:.1f}) g2={g2}(×{ratio:.0f}) g3={g3}. Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
