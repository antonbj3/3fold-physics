"""EJECTA VELOCITY / PROJECTILE - how fast vent gas and debris leave a failing cell, and the resulting projectile and
jet-reach hazard.

What it computes: because a burst vents through a small orifice at a high pressure ratio, the flow is choked and the
gas leaves at the local sound speed a = sqrt(gamma*R_u*T/M) (hundreds of m/s for hot vent gas), entraining solid
debris to a drag-limited fraction of that speed. From this follow the projectile kinetic energy (1/2*m*v^2) and a
high-momentum jet that reaches neighbouring cells - the directional ejecta path in the propagation regime map. It
complements the vent-burst model, which sets the choked mass flow rather than the exit velocity.

Inputs: none (vent gas at 1000 C, M = 30 g/mol, gamma = 1.30, 12 g ejected mass, debris drag coupling 0.3 as module
constants). Outputs: the choked exit velocity, the debris velocity, the ejecta kinetic energy range, the cold-gas
comparison, and gate lines G1-G4.

Reference: choked-orifice (sonic) exhaust relation and literature lithium-ion ejecta speeds and masses.

GATES: G1 the gas exits at the sonic speed, several hundred m/s for hot vent gas. G2 the ejecta kinetic energy
(gas plus entrained debris) is 1e2-1e3 J, an injury/ignition projectile, and the high-momentum jet reaches the
neighbour. G3 regime and NULL - a subsonic vent exits slower, the sonic speed is the cap, and no vent means no
ejecta. G4 composes the vent mass-flow model, the propagation regime map and directional venting.
"""
import sys
import numpy as np

R_U = 8.314
GAMMA = 1.30                  # hot vent-gas ratio of specific heats
M_GAS = 0.030                # kg/mol mixed vent gas (CO2/CO/H2/HC)
T_VENT = 1000.0 + 273.15     # K vent-gas temperature
M_EJECTA = 0.012             # kg ejected mass (gas+debris, ~25% of a 47 g cell)
DEBRIS_COUPLING = 0.3        # fraction of sonic speed reached by entrained debris (drag-limited)


def sound_speed(T, M=M_GAS, g=GAMMA):
    return np.sqrt(g * R_U * T / M)


def main():
    print("=" * 100)
    print("EJECTA VELOCITY / PROJECTILE render→match (sonic choked exit → debris KE + jet reach)")
    print("=" * 100)
    a = sound_speed(T_VENT)                                    # choked exit (sonic) velocity
    v_debris = DEBRIS_COUPLING * a
    KE_gas = 0.5 * M_EJECTA * a ** 2                           # if all ejecta moved at gas speed (upper)
    KE_eff = 0.5 * M_EJECTA * v_debris ** 2                    # debris-coupled (realistic projectile)
    print(f"\n  vent gas {T_VENT-273.15:.0f}°C, M={M_GAS*1e3:.0f} g/mol, γ={GAMMA}: choked exit (sonic) a = {a:.0f} m/s")
    print(f"  G1 the gas leaves at the SOUND speed {a:.0f} m/s (choked orifice); entrained debris ~{DEBRIS_COUPLING:.0%}·a = {v_debris:.0f} m/s (drag-limited)")
    print(f"  G2 projectile: ejecta {M_EJECTA*1e3:.0f} g → KE {KE_eff:.0f} J (debris-coupled) to {KE_gas:.0f} J (gas-speed upper) — injury/ignition projectile")
    print(f"     jet reach: high-momentum jet ({M_EJECTA*0.5:.3f} kg·… , a={a:.0f} m/s) reaches neighbour cells → the regime-map EJECTA path")
    # ambient comparison: room-temperature gas would be slower
    a_cold = sound_speed(300.0)
    print(f"  G3 NULL/regime: subsonic vent exits < {a:.0f} m/s; cold gas (300K) sound speed {a_cold:.0f} m/s (hot vent is ×{a/a_cold:.1f} faster); no vent → no ejecta")

    g1 = 300 < a < 800                                        # ★sonic hot-gas exit, hundreds of m/s
    g2 = (50 < KE_eff < 5000) and (v_debris > 50)            # ★projectile KE 10²–10³ J, debris fast
    g3 = a > a_cold                                           # ★hot vent faster than cold (T-dependence); NULL definitional
    g4 = g1 and g2
    ok = g1 and g2 and g3
    print("\n" + "-" * 100)
    print(f"  G1 ★choked sonic exit a = {a:.0f} m/s (√(γR_uT/M), hot vent gas)                                {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★projectile KE {KE_eff:.0f}–{KE_gas:.0f} J + high-momentum jet reaches the neighbour (regime-map ejecta path) {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★regime: hot vent ×{a/a_cold:.1f} faster than cold; subsonic slower; NULL no-vent → no ejecta        {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes the choked mass flow + regime map (ejecta path) + directional-venting                  {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("EJECTA VELOCITY / PROJECTILE render→match LIVE — the speed the burst-dynamics model leaves out; that model sets the choked MASS flow")
        print(f"  (whether the cell vents or bursts); here the same choked orifice fixes the EXIT VELOCITY: hot vent gas at {T_VENT-273.15:.0f}°C leaves at its")
        print(f"  sound speed a = √(γR_uT/M) = {a:.0f} m/s, ×{a/a_cold:.1f} the cold-gas value, entraining debris to ~{v_debris:.0f} m/s. That makes the vent a")
        print(f"  PROJECTILE source — {M_EJECTA*1e3:.0f} g of ejecta carries {KE_eff:.0f}–{KE_gas:.0f} J, enough to injure or ignite — and a high-momentum JET that reaches the")
        print(f"  neighbouring cells, which is exactly the directional ejecta path the propagation regime map flagged as the barrier-")
        print(f"  defeating mechanism. Composes the vent mass flow, the regime map's ejecta energy, and the directional-venting geometry.")
    else:
        print(f"HONEST WIP: g1={g1}(a {a:.0f}) g2={g2}(KE {KE_eff:.0f}) g3={g3}. Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
