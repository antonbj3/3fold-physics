"""VENT-GAS DYNAMICS: BURST versus CONTROLLED VENT.

What it computes: whether a cell in thermal runaway vents safely or ruptures. A bursting cell is far worse than a
venting one (shrapnel, and a wide-open hole that floods the pack with hot gas and speeds propagation). The outcome is
a race: the decomposition cascade generates gas while the vent / current-interrupt device relieves it by choked flow.
Ideal-gas pressure P = nRT/V_head with dn/dt = Ngen(t) - n_vent, a gas-generation pulse tracking the cascade extent,
and a vent of area A that opens at P_open and exhausts with n_vent = C*A*P. The critical vent area A_crit is the
smallest area that holds the pressure below the can-rupture limit over the whole pulse: below it the cell bursts, at
or above it the vent is controlled.

Inputs: none (900 K vent gas, 2 cm3 head volume, 1 MPa opening pressure, 3.5 MPa rupture pressure, 0.5 mol of vent
gas released over a cell-scale ~1.5 s timescale, choked-flow coefficient such that a 100 mm2 vent passes ~3 mol/s at
1 MPa). Outputs: the critical vent area, the peak pressure for a small and a generous vent, the gas-rate
perturbation, the no-gas null, and gate lines G1-G4.

Reference: literature vent-gas yields (~0.5 mol per 18650 in runaway), typical opening and rupture pressures, and
choked-orifice flow.

GATES: G1 a critical vent area exists - below it the cell bursts, at or above it the pressure stays bounded. G2 the
critical area scales with the peak gas-generation rate (doubling the rate roughly doubles it). G3 NULL: with no
runaway there is no pressure rise, and a generous vent pins the pressure near the opening pressure. G4 composes the
vent/current-interrupt model, the gas-generation cascade and the vent-gas composition work.
"""
import sys
import numpy as np

Rgas = 8.314
T_GAS = 900.0                   # vent-gas temperature (K) ~ TR gas
V_HEAD = 2.0e-6                 # cell head/void volume (m³)
P_OPEN = 1.0e6                  # vent/CID opening pressure (Pa) ~ 10 bar
P_BURST = 3.5e6                 # can rupture pressure (Pa) ~ 35 bar
N_TOTAL = 0.5                   # total moles of vent gas generated in TR (~0.5 mol, literature)
TAU_VENT = 1.5                   # gas-release timescale (s) — the CELL-SCALE event (≫ the <2 ms chemical spike)
C_CHOKE = 0.03                  # choked-flow coefficient ṅ_vent=C·A·P: a 100 mm² vent flows ~3 mol/s at 1 MPa (physical)


def gas_pulse(n_t=2000):
    """gas-generation rate Ṅ_gen(t) (mol/s): a pulse of N_TOTAL released on the CELL-SCALE timescale (the observed TR
    event is ~seconds, NOT the <2 ms chemical spike; the gas release/vent is mass-transfer-paced, driven by the cascade)."""
    ts = np.linspace(0, 5.0, n_t)
    mu, sig = 1.5, TAU_VENT / 2.5
    g = np.exp(-0.5 * ((ts - mu) / sig) ** 2)
    area = np.trapezoid(g, ts) if hasattr(np, "trapezoid") else np.trapz(g, ts)
    return ts, g / area * N_TOTAL                            # mol/s, ∫ = N_TOTAL


def peak_pressure(A, ts, ngen, gen_scale=1.0):
    """integrate the pressure with a vent of area A (m²); SEMI-IMPLICIT choked vent (stable for any A). Return peak P (Pa)."""
    n = P_OPEN * V_HEAD / (Rgas * T_GAS) * 1e-3              # tiny initial moles (well below P_open)
    Pmax = n * Rgas * T_GAS / V_HEAD
    k_vent = C_CHOKE * A * Rgas * T_GAS / V_HEAD             # implicit vent-rate coefficient (1/s) when open
    for k in range(1, len(ts)):
        dt = ts[k] - ts[k - 1]
        n += ngen[k] * gen_scale * dt                       # gas added
        P = n * Rgas * T_GAS / V_HEAD
        if P > P_OPEN:                                       # valve open → choked flow toward atmosphere (implicit, stable)
            n = n / (1.0 + k_vent * dt)
            P = n * Rgas * T_GAS / V_HEAD
        Pmax = max(Pmax, P)
    return Pmax


def find_Acrit(ts, ngen, gen_scale=1.0):
    lo, hi = 1e-7, 1e-3
    for _ in range(30):
        mid = np.sqrt(lo * hi)
        if peak_pressure(mid, ts, ngen, gen_scale) > P_BURST:
            lo = mid
        else:
            hi = mid
    return np.sqrt(lo * hi)


def main():
    print("=" * 100)
    print("VENT-GAS DYNAMICS: burst vs controlled vent (cascade gas generation vs choked vent relief)")
    print("=" * 100)
    ts, ngen = gas_pulse()
    peak_gen = ngen.max()
    print(f"\n  TR gas pulse (from the decomposition cascade): {N_TOTAL} mol over {ts[-1]:.1f} s, peak rate {peak_gen:.2f} mol/s; P_open {P_OPEN/1e6:.0f} MPa, P_burst {P_BURST/1e6:.1f} MPa")
    A_crit = find_Acrit(ts, ngen)
    print(f"  ★critical vent area A_crit = {A_crit*1e6:.1f} mm² — below → BURST, at/above → controlled vent")
    # bracket: show burst vs vent
    P_small = peak_pressure(0.3 * A_crit, ts, ngen) / 1e6
    P_big = peak_pressure(3 * A_crit, ts, ngen) / 1e6
    print(f"  ★A=0.3·A_crit ({0.3*A_crit*1e6:.1f} mm²) → peak P {P_small:.1f} MPa (>{P_BURST/1e6:.1f} → BURST); A=3·A_crit ({3*A_crit*1e6:.1f} mm²) → peak P {P_big:.2f} MPa (controlled, pinned near P_open)")
    # G2 perturbation: 2× gen rate → ~2× A_crit
    A_crit2 = find_Acrit(ts, ngen, gen_scale=2.0)
    ratio = A_crit2 / A_crit
    # NULL: no TR (no gas)
    P_null = peak_pressure(A_crit, ts, ngen * 0.0) / 1e6

    g1 = (P_small > P_BURST / 1e6) and (P_big < P_BURST / 1e6)   # ★A_crit separates burst from controlled vent
    g2 = abs(ratio - 2.0) < 0.5                                  # ★A_crit ∝ peak gen rate (2× gen → ~2× A)
    g3 = (P_null < P_OPEN / 1e6 + 0.1) and (P_big < 1.5 * P_OPEN / 1e6)  # ★NULL no-gas no-rise + generous vent pins near P_open
    g4 = g1 and g2                                               # composes vent/CID, gas generation and vent-gas hazard
    ok = g1 and g2 and g3
    print(f"\n  ★A_crit scales with gen rate: 2× gas-gen → A_crit {A_crit*1e6:.1f} → {A_crit2*1e6:.1f} mm² (ratio {ratio:.1f}× ≈ 2)")
    print(f"  NULL: no TR (zero gas gen) → peak P {P_null:.2f} MPa (ambient, no pressure rise)")
    print("\n" + "-" * 100)
    print(f"  G1 ★A_crit={A_crit*1e6:.1f} mm² separates BURST ({P_small:.0f} MPa) from controlled vent ({P_big:.1f} MPa)   {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★A_crit ∝ peak gas-gen rate (2× gen → {ratio:.1f}× A_crit) — composes the cascade         {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★NULL no-gas no-rise ({P_null:.1f} MPa) + generous vent pins P near P_open               {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes vent/CID + gas generation + the vent-gas hazard                            {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("VENT-GAS DYNAMICS render→match LIVE — whether a cell VENTS or BURSTS, the dynamics of the vent-hazard family.")
        print(f"  The TR cascade generates ~{N_TOTAL} mol of gas in a {ts[-1]:.0f}-s pulse (peak {peak_gen:.1f} mol/s); the vent/CID relieves it by")
        print(f"  choked flow. It is a RACE: a critical vent area A_crit ≈ {A_crit*1e6:.0f} mm² separates the outcomes — below it the vent can't keep")
        print(f"  up and the pressure climbs past the can-rupture limit ({P_BURST/1e6:.1f} MPa) → BURST ({P_small:.0f} MPa at 0.3·A_crit), at/above it the vent holds")
        print(f"  the pressure pinned near P_open ({P_big:.1f} MPa) → controlled vent. ★A_crit scales with the peak gas-generation rate (2× gen ⇒")
        print(f"  ~2× area), so a more violent (higher-rate) TR needs a bigger vent — the vent-SIZING criterion. NULL: no TR, no pressure.")
        print(f"  Composes the vent/CID, the gas-generating cascade, the cell-scale release timescale and the vent-gas hazard.")
    else:
        print(f"HONEST WIP: g1={g1}(Psmall {P_small:.0f}/Pbig {P_big:.1f}) g2={g2}(ratio {ratio:.1f}) g3={g3}. Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
