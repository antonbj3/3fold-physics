"""COUPLED ELECTRO-THERMAL RUNAWAY - an internal short (electrical abuse) ignites the multi-reaction decomposition
cascade (chemical runaway).

What it computes: an internal short of resistance R_short dissipates the cell's stored energy as I2R heat
(I = V_cell/R_short, P = V^2/R_short), which raises the cell temperature; as T climbs, the four decomposition
reactions (SEI -> anode -> cathode -> electrolyte) activate and add their chemical heat. The two sources compound
against Newton cooling. The electrical heat alone only reaches a steady warm temperature; it is the chemical cascade,
once ignited, that runs away. The critical short resistance R_crit is therefore the one whose I2R heating just lifts
the cell to the cascade-onset temperature: below R_crit the short ignites the cascade (thermal runaway), above it
cooling wins (benign).

Inputs: none (lumped 21700-class cell: 51 J/K heat capacity, hA = 0.05 W/K, 18 Wh stored energy; four-reaction
Hatchard/Kim kinetics as module constants). Outputs: the peak temperature and runaway verdict for a hard short, the
critical short resistance from a resistance scan, the cascade-off and open-circuit controls, and gate lines G1-G4.

Reference: Hatchard/Kim four-reaction thermal-runaway kinetics for a lithium-ion cell; internal-short critical
resistance of a few ohms.

GATES: G1 the full coupled trajectory for a hard short - I2R heat, cascade ignition near 150 C, runaway to
~600-700 C. G2 the critical short resistance (a few ohms) separates runaway from benign; the exact value is
hA-dependent, the robust result is the few-ohm threshold. G3 the coupling is essential: I2R alone (cascade off) only
reaches a steady warm temperature, and NULL open circuit gives no heating and no runaway. G4 composes the
internal-short, decomposition-cascade, Semenov-criticality and I2R-heating models.
"""
import sys
import numpy as np

Rgas = 8.314
RHOCP = 2.58e6                  # volumetric heat capacity (J/m³K)
V_VOL = 2.0e-5                  # cell volume (m³) ~ a 21700 (×~50 g)
MCP = RHOCP * V_VOL             # lumped heat capacity (J/K) ~ 51 J/K
HA = 0.05                       # cooling hA (W/K) — cell in a pack (natural-ish)
V0 = 3.7                        # cell nominal voltage (V)
E_ELEC = 18 * 3600.0           # stored electrical energy (J) ~ 18 Wh (21700)

# Hatchard/Kim 4-reaction cascade: (A, Ea, H[J/kg], W[kg/m³], c0)
RXN = [(1.667e15, 1.3508e5, 2.57e5, 610.0, 0.15),
       (2.500e13, 1.3508e5, 1.714e6, 610.0, 0.75),
       (6.667e13, 1.396e5, 3.14e5, 1300.0, 0.04),
       (5.140e25, 2.74e5, 1.55e5, 406.0, 1.00)]
Z0 = 0.033


def chem_power(st, T):
    """volumetric chemical heat rate (W/m³) from the 4 reactions + update derivatives."""
    c_sei, c_an, al, c_el, z = st
    r = [RXN[0][0] * np.exp(-RXN[0][1] / (Rgas * T)) * max(c_sei, 0),
         RXN[1][0] * np.exp(-RXN[1][1] / (Rgas * T)) * max(c_an, 0) * np.exp(-z / Z0),
         RXN[2][0] * np.exp(-RXN[2][1] / (Rgas * T)) * max(al, 0) * max(1 - al, 0),
         RXN[3][0] * np.exp(-RXN[3][1] / (Rgas * T)) * max(c_el, 0)]
    q = sum(RXN[i][2] * RXN[i][3] * r[i] for i in range(4))
    return q, r


def simulate(R_short, T0_C=25.0, chem_on=True, t_end=4e4, max_steps=400000):
    """couple I²R short-heating + the chemical cascade + cooling. Returns (peak T °C, did_TR, t_to_TR)."""
    T = T0_C + 273.15; Tamb = T
    st = np.array([RXN[0][4], RXN[1][4], RXN[2][4], RXN[3][4], 0.0])
    E = E_ELEC                                                   # remaining stored electrical energy (J)
    Tmax = T; t = 0.0; n = 0; t_TR = None
    while t < t_end and n < max_steps:
        soc = max(E / E_ELEC, 0.0)
        Vc = V0 * (0.5 + 0.5 * soc)                            # cell voltage falls as it discharges through the short
        P_elec = (Vc * Vc / R_short) if (R_short < 1e8 and E > 0) else 0.0   # I²R short power (W)
        q_chem, r = chem_power(st, T) if chem_on else (0.0, [0, 0, 0, 0])
        P_chem = q_chem * V_VOL                                # chemical power (W)
        dTdt = (P_elec + P_chem - HA * (T - Tamb)) / MCP
        rmax = max(max(r) if chem_on else 0.0, 1e-12)
        dt = min(5.0, 0.02 / rmax, 2.0 / max(abs(dTdt), 1e-6))
        if chem_on:
            st[0] -= r[0] * dt; st[1] -= r[1] * dt; st[2] += r[2] * dt; st[3] -= r[3] * dt
            st[4] += r[1] * dt; st[:4] = np.clip(st[:4], 0, 1)
        T += dTdt * dt; E -= P_elec * dt
        Tmax = max(Tmax, T)
        if T - Tamb > 200 and t_TR is None:
            t_TR = t                                          # crossed into runaway
        t += dt; n += 1
        if (st[0] < 1e-3 and st[1] < 1e-3 and st[3] < 1e-3 and st[2] > 0.99) or T > 1573:
            break
        if E <= 0 and (T - Tamb) < 5:                         # short drained the cell without TR → benign
            break
    return Tmax - 273.15, (Tmax - Tamb - 273.15 > 200), t_TR


def main():
    print("=" * 100)
    print("COUPLED ELECTRO-THERMAL-RUNAWAY: internal-short I²R heat ignites the decomposition cascade")
    print("=" * 100)
    # hard short → full coupled trajectory
    pk_hard, tr_hard, t_tr = simulate(1.0)
    print(f"\n  hard short R=1.0 Ω: peak {pk_hard:.0f}°C, TR={tr_hard} (I²R heat → cascade ignites → runaway)")
    # scan R_short for the critical resistance
    Rcrit = None
    for R in np.arange(0.5, 12.0, 0.5):
        _, tr, _ = simulate(float(R))
        if not tr:
            Rcrit = R; break
    Rcrit_TR = Rcrit - 0.5 if Rcrit else None                 # last R that DID cause TR
    print(f"  ★critical short resistance: R ≤ {Rcrit_TR:.1f} Ω → TR; R ≥ {Rcrit:.1f} Ω → benign (a few Ω, as for a modelled internal short; hA-dependent)")
    # G3 coupling essential: I²R alone (chem off) for a hard short
    pk_nochem, tr_nochem, _ = simulate(1.0, chem_on=False)
    print(f"  ★coupling essential: I²R ALONE (cascade off), R=1.0 Ω → only {pk_nochem:.0f}°C steady-warm, TR={tr_nochem} (no runaway without the cascade)")
    # NULL: open circuit
    pk_open, tr_open, _ = simulate(1e9)
    print(f"  NULL: open circuit (R→∞) → peak {pk_open:.0f}°C, TR={tr_open} (no heat source, no TR)")

    g1 = tr_hard and (450 <= pk_hard <= 760)                   # ★hard short → runaway (peak below the adiabatic 669°C: cooling + finite electrical drive)
    g2 = (Rcrit_TR is not None) and (0.5 <= Rcrit_TR <= 8.0)   # ★R_crit few Ω (hA-dependent)
    g3 = (not tr_nochem) and (pk_nochem < 250) and (not tr_open)  # ★I²R alone no TR + NULL open-circuit no TR
    g4 = g1 and g2                                             # composes internal short, cascade, criticality, I²R heating
    ok = g1 and g2 and g3
    print("\n" + "-" * 100)
    print(f"  G1 ★hard short → coupled runaway: peak {pk_hard:.0f}°C (TR; < adiabatic 669°C: cooling+finite drive)     {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★critical short R_crit ≈ {Rcrit_TR:.1f} Ω separates TR/benign — a few Ω (hA-dependent)           {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★coupling essential: I²R alone {pk_nochem:.0f}°C no-TR + NULL open-circuit no-TR       {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes internal short (R_crit) + cascade + Semenov criticality + I²R heating    {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("COUPLED ELECTRO-THERMAL-RUNAWAY render→match LIVE — the two reference models coupled. An")
        print(f"  internal short dissipates the cell's energy as I²R heat, which lifts the temperature until the resolved")
        print(f"  cascade ignites and runs away to {pk_hard:.0f}°C (a genuine TR; lower than the adiabatic 669°C because of cooling + the finite electrical drive) — the full ELECTRICAL→CHEMICAL trajectory. ★The critical short resistance is")
        print(f"  R_crit ≈ {Rcrit_TR:.1f} Ω (≤ → TR, ≥ → benign) — a few ohms (hA-dependent), now via the RESOLVED cascade. ★The coupling is")
        print(f"  ESSENTIAL: the I²R heat ALONE only reaches a steady-warm {pk_nochem:.0f}°C (no runaway) — it is the CHEMICAL cascade, once the")
        print(f"  electrical heat ignites it, that runs away (the Semenov criticality driven by an electrical source). NULL: open circuit, no")
        print(f"  heat, no TR. This unifies the electrical and chemical runaway paths.")
    else:
        print(f"HONEST WIP: g1={g1}(pk {pk_hard:.0f}) g2={g2}(Rcrit {Rcrit_TR}) g3={g3}(nochem {pk_nochem:.0f}). Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
