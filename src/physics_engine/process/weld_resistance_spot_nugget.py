#!/usr/bin/env python
"""RESISTANCE SPOT WELD NUGGET — nugget size from a Kohlrausch-Holm-anchored energy balance vs AWS tables.

The contact-melting law is the Kohlrausch φ-θ theorem T_m² − T_b² = U²/(8L) (Lorenz number L=2.44e-8 V²/K²),
anchored on Holm's measured Al melting voltage 0.30–0.45 V. This module reparametrizes it for steel resistance
spot welding and adds the nugget-growth energy balance, gated against the AWS/RWMA sheet-steel rules.

MODEL (derive from geometry)
  Heat: Q = I²·R_dyn·τ (R_dyn = 100 µΩ central, literature dynamic-resistance band 80–150 µΩ for
  steel RSW, Dickinson 1980); nugget fraction η = 0.15 (literature 0.10–0.25; electrodes are
  water-cooled). Nugget = oblate ellipsoid, height h = 1.4·t (penetration ~70% per sheet),
  volume V = (π/6)d²h, melt enthalpy e_v = ρ(cp(Tm−T0)+Lf).
  AWS anchor: minimum nugget diameter d_min = 4√t, typical target 5√t (AWS D8.9M weld-button rule).

PRE-REGISTERED GATES
  G1 φ-θ REPARAM: Al melting voltage from T_m²−T_b²=U²/8L lands in Holm's
     measured 0.30–0.45 V. Fe reported under BOTH U-conventions
     (8L: ~0.79 V, 4L: ~0.56 V) vs Holm's published ~0.6 V — the factor-2 U-convention spread is
     flagged, not hidden; no gate on Fe (convention-ambiguous), Al is the anchored gate.
  G2 AWS NUGGET BAND, 2 thicknesses (diverse instances): standard schedules (t=1.0 mm: 8.0 kA,
     10 cycles=0.167 s; t=1.5 mm: 10.5 kA, 15 cycles=0.25 s; RWMA-class) → predicted d within
     [4√t, 6√t] mm for BOTH, at central R_dyn/η; the η∈[0.10,0.25] band is reported.
  G3 KNOWN-BAD + DIRECTION: 30% undercurrent (0.7·I → 0.49·Q) drives d below the AWS minimum
     4√t at both thicknesses (undersized-weld detected) AND d is monotone in I.

I/O: no input files; writes the gate record to artifacts/weld_resistance_spot_nugget.json and prints a PASS/FAIL
verdict (exit 0 on pass). Anchors: Holm, "Electric Contacts"; AWS D8.9M weld-button rule; Dickinson 1980
dynamic-resistance data; RWMA-class schedules.
"""
import json, os, sys
import numpy as np

L_LORENZ = 2.44e-8
STEEL = dict(Tm=1793.0, T0=293.0, rho=7200.0, cp=650.0, Lf=270e3)
AL = dict(Tm=933.0, T0=293.0)
R_DYN = 100e-6                # ohm, central (lit 80-150 uΩ)
ETA = 0.15                    # nugget heat fraction, central (lit 0.10-0.25)
SCHEDULES = {1.0: dict(I=8000.0, tau=0.167), 1.5: dict(I=10500.0, tau=0.250)}   # RWMA-class sheet-steel


def u_melt(mat, factor=8.0):
    return float(np.sqrt(factor * L_LORENZ * (mat["Tm"] ** 2 - mat["T0"] ** 2)))


def nugget_d(I, tau, t_mm, eta=ETA, r_dyn=R_DYN):
    Q = I ** 2 * r_dyn * tau * eta
    e_v = STEEL["rho"] * (STEEL["cp"] * (STEEL["Tm"] - STEEL["T0"]) + STEEL["Lf"])
    V = Q / e_v
    h = 1.4 * t_mm * 1e-3
    d = np.sqrt(6.0 * V / (np.pi * h))
    return float(d * 1000.0)  # mm


def main():
    print("=" * 100)
    print("RESISTANCE SPOT WELD NUGGET — Kohlrausch-Holm-anchored energy balance vs AWS 4√t")
    print("=" * 100)
    ua8 = u_melt(AL, 8.0); uf8 = u_melt(STEEL, 8.0); uf4 = u_melt(STEEL, 4.0)
    g1 = bool(0.30 <= ua8 <= 0.45)
    print(f"[G1] φ-θ (8L convention): U_melt Al={ua8:.3f} V vs Holm 0.30–0.45 -> {g1} | "
          f"Fe: 8L={uf8:.3f} / 4L={uf4:.3f} V vs published ~0.6 (convention spread flagged, not gated)")

    rows = {}
    g2_all, g3_all = [], []
    for t, sch in SCHEDULES.items():
        d = nugget_d(sch["I"], sch["tau"], t)
        lo, hi = 4 * np.sqrt(t), 6 * np.sqrt(t)
        band = [nugget_d(sch["I"], sch["tau"], t, eta=e) for e in (0.10, 0.25)]
        under = nugget_d(0.7 * sch["I"], sch["tau"], t)
        mono = bool(nugget_d(0.9 * sch["I"], sch["tau"], t) < d < nugget_d(1.1 * sch["I"], sch["tau"], t))
        ok2 = bool(lo <= d <= hi); ok3 = bool(under < lo and mono)
        g2_all.append(ok2); g3_all.append(ok3)
        rows[t] = dict(I_kA=sch["I"] / 1e3, tau_s=sch["tau"], d_mm=round(d, 2), aws_min=round(lo, 2),
                       band_6rt=round(hi, 2), eta_band=[round(b, 2) for b in band],
                       undercurrent_d=round(under, 2), in_band=ok2, undersized_detected=ok3)
        print(f"[G2/G3] t={t}mm ({sch['I']/1e3:.1f} kA, {sch['tau']}s): d={d:.2f} mm vs AWS [4√t,6√t]=[{lo:.2f},{hi:.2f}] "
              f"(η-band {rows[t]['eta_band']}) -> {ok2} | 0.7·I -> d={under:.2f} < {lo:.2f} (undersized detected) + monotone -> {ok3}")
    g2, g3 = all(g2_all), all(g3_all)

    ok = g1 and g2 and g3
    outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(outdir, exist_ok=True)
    json.dump({
        "claim": ("weld_resistance_spot_nugget: the Kohlrausch-Holm φ-θ law (Al melting-voltage anchor reproduced) "
                  "+ an energy-balance nugget model "
                  "with pre-registered literature R_dyn/η reproduces the AWS/RWMA nugget-size band [4√t,6√t] at two "
                  "sheet thicknesses under standard schedules, and detects the known-bad 30% undercurrent as an "
                  "undersized weld (below the AWS 4√t minimum) with monotone current response."),
        "gates": {"G1_U_melt_Al_V": round(ua8, 3), "G1_holm_band": [0.30, 0.45],
                  "G1_U_melt_Fe_8L": round(uf8, 3), "G1_U_melt_Fe_4L": round(uf4, 3),
                  "G1_Fe_published": 0.6, "G1": g1, "G2_G3_rows": rows, "G2": g2, "G3": g3,
                  "verdict": "PASS" if ok else "FAIL"},
        "provenance": ("phi-theta T_m²−T_b²=U²/8L, L=2.44e-8 (Holm Al anchor); nugget energy balance Q=I²·R_dyn·τ·η, R_dyn=100 µΩ (lit 80–150, Dickinson 1980), "
                       "η=0.15 (lit 0.10–0.25, reported band), ellipsoid h=1.4t; AWS D8.9M weld-button rule d_min=4√t; "
                       "RWMA-class schedules 1.0mm/8kA/10cyc, 1.5mm/10.5kA/15cyc; Fe U-convention factor-2 spread "
                       "flagged; CPU"),
    }, open(os.path.join(outdir, "weld_resistance_spot_nugget.json"), "w"), indent=1)
    print("=" * 100)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — G1(Holm-Al)={g1} G2(AWS-band 2t)={g2} G3(undersized-detected)={g3}")
    print("EVIDENCE -> artifacts/weld_resistance_spot_nugget.json")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
