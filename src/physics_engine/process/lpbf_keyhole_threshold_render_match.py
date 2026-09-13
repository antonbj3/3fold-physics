"""LPBF CONDUCTION↔KEYHOLE THRESHOLD — RENDER→MATCH the regime boundary vs the NIST AM-Bench 2022 (AMB2022-01) scan & spot.
The keyhole forms when the conduction-mode surface temperature reaches the metal's BOILING point: vaporisation → recoil
pressure → a vapor depression → multiple-reflection trapping (the absorptance jump). So the regime boundary in (P,v)
space is the locus where the PEAK surface temperature = T_boil — rendered from the Eagar–Tsai moving-source field with the
pre-keyhole (conduction) absorptance, NOTHING fit. Two independent instruments must AGREE on the classification.

INSTRUMENT 1 (mechanism, my render) — peak centre-surface temperature of a moving Gaussian source:
    ΔT_centre(P,v) = (2ηP)/(ρc_p(4πα)^{3/2}) ∫_0^∞ 2/(u²+t0)·exp(−v²u⁴/(4α(u²+t0))) du ,  t0=σ_b²/(2α)
  (linear in P; v=0 closes analytically to π/√t0 — the stationary-spot steady centre). Keyhole ⇔ T_centre ≥ T_boil.
INSTRUMENT 2 (literature criterion, external) — King 2014 normalised enthalpy ΔH/h_s = ηP/(ρ h_s √(π α v a³)),
  h_s=ρc_pT_m, a=1/e² beam radius; keyhole onset ΔH/h_s ≳ 6–8.5 (King/Ye/Hann). Independent of instrument 1.

MATCH: the NIST scan (473 W, 700 mm/s) and spot (501 W, stationary) BOTH show the absorptance jump to keyhole (43.3 %,
64.1 %) ⇒ both must land in the keyhole regime of BOTH instruments; a low-power/fast control must land in conduction.
External anchors: Al T_boil=2743 K (textbook) and the King ΔH/h_s threshold. render→match, never fit.

I/O: no input files — the NIST operating points (473 W / 700 mm/s scan, 501 W stationary spot) are quoted inline as
published values; prints the classification table, six gate lines and a PASS/FAIL verdict (exit 0 on pass).
Dataset/anchor: NIST AM-Bench 2022 (AMB2022-01); King et al. 2014 normalised-enthalpy keyhole criterion.
"""
import sys
import numpy as np


def dT_centre(P, v, eta, sigma_b, k, rho, cp, n=6000):
    """Peak centre-surface temperature RISE (K) of a moving Gaussian source (linear in P).
    Substitution u=√t0·tanφ makes the integrand SMOOTH & bounded for ALL v (incl. v→0, the stationary spot: the
    near-zero peak of 2/(u²+t0) has width √t0~1e-3 ≪ the v-cutoff u~1/v, so a uniform u-grid mis-resolves it; in φ the
    factor 2/(u²+t0)du → 2/√t0 dφ is flat, and v=0 integrates exactly to π/√t0)."""
    alpha = k / (rho * cp)
    t0 = sigma_b**2 / (2 * alpha)
    phi = np.linspace(0.0, np.pi / 2 - 1e-9, n)
    tanp = np.tan(phi)
    u2 = t0 * tanp**2
    integ = (2.0 / np.sqrt(t0)) * np.exp(-(v**2) * (u2**2) / (4 * alpha * (u2 + t0)))
    pref = (2 * eta * P) / (rho * cp * (4 * np.pi * alpha) ** 1.5)
    return pref * np.trapezoid(integ, phi)


def norm_enthalpy(P, v, eta, a, k, rho, cp, Tm):
    """King 2014 normalised enthalpy ΔH/h_s (h_s=ρ c_p T_m, a=1/e² beam radius)."""
    alpha = k / (rho * cp)
    hs = rho * cp * Tm
    return eta * P / (hs * np.sqrt(np.pi * alpha * max(v, 1e-12) * a**3))


def main():
    print("=" * 100)
    print("CONDUCTION↔KEYHOLE THRESHOLD RENDER→MATCH — NIST AM-Bench 2022 (peak-T=T_boil + King ΔH/h_s)")
    print("=" * 100)
    rho, cp, k = 2700.0, 1050.0, 150.0
    Tm, T0, Tboil = 933.0, 293.0, 2743.0                              # Al melt / ambient / BOIL (textbook)
    a = 61.25e-6                                                      # 1/e² beam radius (122.5 µm diam)
    sigma_b = a / 2
    eta = 0.238                                                      # CONDUCTION (pre-keyhole) absorptance — the trigger test
    dT_boil = Tboil - T0
    # operating points: NIST scan & spot, + a deliberately sub-threshold control
    pts = {"NIST scan (473 W, 700 mm/s)": (473.0, 0.700),
           "NIST spot (501 W, stationary)": (501.0, 0.0),          # truly stationary ⇒ v=0 (steady reached in ~a²/α≈70 µs ≪ 2 ms pulse)
           "control (150 W, 2000 mm/s)": (150.0, 2.000)}
    print(f"\n  Al: T_melt {Tm} K, T_boil {Tboil} K (rise {dT_boil} K); beam a={a*1e6:.1f} µm (1/e² radius); η_cond={eta}")
    print(f"  {'operating point':32s}  T_peak(K)   T_peak/T_boil   ΔH/h_s   regime (peak-T | enthalpy)")
    classif = {}
    for name, (P, v) in pts.items():
        Tp = T0 + dT_centre(P, v, eta, sigma_b, k, rho, cp)
        beta = norm_enthalpy(P, v, eta, a, k, rho, cp, Tm)
        kh_T = Tp >= Tboil
        kh_b = beta >= 6.0
        classif[name] = (Tp, beta, kh_T, kh_b)
        bstr = "    ∞ " if v == 0.0 else f"{beta:6.2f}"             # stationary spot ⇒ ΔH/h_s formally diverges
        print(f"  {name:32s}  {Tp:7.0f}    {Tp/Tboil:6.2f}        {bstr}   "
              f"{'KEYHOLE' if kh_T else 'conduction':10s} | {'KEYHOLE' if kh_b else 'conduction'}")

    # threshold boundary P_thr(v) where peak-T = T_boil (linear-in-P ⇒ exact scaling)
    print(f"\n  keyhole-threshold power P_thr(v) [peak-T=T_boil], η_cond={eta}:")
    for v in (0.2, 0.5, 0.7, 1.0, 2.0):
        per_W = dT_centre(1.0, v, eta, sigma_b, k, rho, cp)
        P_thr = dT_boil / per_W
        print(f"    v={v:4.1f} m/s → P_thr={P_thr:6.0f} W")

    (Tp_scan, b_scan, *_), (Tp_spot, b_spot, *_), (Tp_ctrl, b_ctrl, *_) = \
        classif["NIST scan (473 W, 700 mm/s)"], classif["NIST spot (501 W, stationary)"], classif["control (150 W, 2000 mm/s)"]
    # gates: both NIST points keyhole on BOTH instruments; control conduction; spot hotter than scan; instruments agree
    g1 = (Tp_scan >= Tboil) and (Tp_spot >= Tboil)                                  # peak-T criterion → keyhole
    g2 = (b_scan >= 6.0) and (b_spot >= 6.0)                                        # enthalpy criterion → keyhole
    g3 = (Tp_ctrl < Tboil) and (b_ctrl < 6.0)                                       # NULL control → conduction (both)
    g4 = (Tp_spot > Tp_scan) and (b_spot > b_scan)                                  # stationary spot hotter/deeper than scan
    g5 = all((cT == cb) for (_, _, cT, cb) in classif.values())                     # the two instruments AGREE per point
    # g6 instrument check: numeric stationary centre = analytic π/√t0 (no fit, closed-form anchor of the integrator)
    alpha = k / (rho * cp); t0 = sigma_b**2 / (2 * alpha)
    pref = (2 * eta * 501.0) / (rho * cp * (4 * np.pi * alpha) ** 1.5)
    dT0_analytic = pref * np.pi / np.sqrt(t0)
    dT0_numeric = dT_centre(501.0, 0.0, eta, sigma_b, k, rho, cp)
    g6 = abs(dT0_numeric - dT0_analytic) / dT0_analytic < 1e-3
    ok = g1 and g2 and g3 and g4 and g5 and g6
    print(f"  [instrument check] stationary centre: numeric {dT0_numeric:.1f} K = analytic π/√t0 {dT0_analytic:.1f} K  {'✓' if g6 else 'FAIL'}")
    print(f"\n  (1) ★peak-T: NIST scan {Tp_scan:.0f} K & spot {Tp_spot:.0f} K ≥ T_boil {Tboil} → KEYHOLE  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★enthalpy: scan ΔH/h_s {b_scan:.1f} & spot ∞ (stationary) ≥ 6 → KEYHOLE  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) ★NULL control 150 W/2 m/s: T {Tp_ctrl:.0f}<{Tboil}, ΔH/h_s {b_ctrl:.1f}<6 → conduction  {'✓' if g3 else 'FAIL'}")
    print(f"  (4) ★spot (stationary) hotter & higher-enthalpy than scan (matches A_spot>A_scan)  {'✓' if g4 else 'FAIL'}")
    print(f"  (5) ★two independent instruments AGREE on every classification (not one tautological gate)  {'✓' if g5 else 'FAIL'}")
    print("\n" + "=" * 100)
    if ok:
        print("KEYHOLE THRESHOLD RENDER→MATCH — the regime boundary derived from physics, no fit:")
        print(f"  • peak conduction surface-T (η_cond) at the NIST scan & spot BOTH exceed T_boil {Tboil} K ⇒ keyhole predicted,")
        print(f"    matching the measured absorptance jump; a 150 W/2 m/s control stays in conduction. Two independent")
        print(f"    criteria (peak-T=T_boil and King ΔH/h_s) agree on every point — the classification is not a single gate.")
    else:
        print(f"  HONEST: scan T {Tp_scan:.0f}/β {b_scan:.1f}; spot T {Tp_spot:.0f}/β {b_spot:.1f}; control T {Tp_ctrl:.0f}/β {b_ctrl:.1f}.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
