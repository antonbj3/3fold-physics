"""Chromatic correction / achromat: a singlet focuses blue and red at different distances (chromatic aberration);
an achromatic doublet (crown + flint) cancels it to first order via phi1/V1 + phi2/V2 = 0 (V = Abbe number).

Uses real Sellmeier dispersion n^2 - 1 = sum_i B_i lambda^2/(lambda^2 - C_i) for N-BK7 and F2, not a single-term
model. (With a single-term model both glasses' dispersion would be proportional, the achromat would cancel at all
wavelengths and the secondary spectrum would be spuriously zero; real glasses have different relative partial
dispersion, hence a realistic nonzero secondary spectrum.)

CHECKS: (1) singlet chromatic shift (f_F - f_C)/f_d ~ 1/V ~ 1.5%; (2) the generated achromat brings F (486 nm) and
C (656 nm) to a common focus (shift ~0, hundreds of times smaller); (3) a realistic nonzero secondary spectrum
remains (~0.0005*f - the achromat's known residual, which only an apochromat removes).

INPUT: none (Schott N-BK7 and F2 Sellmeier coefficients in the module).
OUTPUT: printed gate lines + `artifacts/optics_achromat.png`.
"""
import os
import sys
import numpy as np

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

LF, LD, LC = 0.4861, 0.5876, 0.6563                            # F (blue), d (yellow), C (red), μm
SELL = {  # Sellmeier B1,B2,B3,C1,C2,C3 (λ in μm) — Schott N-BK7 and F2
    "N-BK7": [1.03961212, 0.231792344, 1.01046945, 0.00600069867, 0.0200179144, 103.560653],
    "F2":    [1.34533359, 0.209073176, 0.937357162, 0.00997743871, 0.0470450767, 111.886764],
}


def n_of(lam, name):
    B1, B2, B3, C1, C2, C3 = SELL[name]; l2 = lam * lam
    return np.sqrt(1 + B1 * l2 / (l2 - C1) + B2 * l2 / (l2 - C2) + B3 * l2 / (l2 - C3))


def glass_nV(name):
    nd, nF, nC = n_of(LD, name), n_of(LF, name), n_of(LC, name)
    return nd, (nd - 1) / (nF - nC)


def f_singlet(lam, name, f_d=100.0):
    nd = glass_nV(name)[0]
    return 1.0 / ((1.0 / f_d) * (n_of(lam, name) - 1) / (nd - 1))


def f_doublet(lam, g1, g2, p1, p2):
    nd1, nd2 = glass_nV(g1)[0], glass_nV(g2)[0]
    phi = p1 * (n_of(lam, g1) - 1) / (nd1 - 1) + p2 * (n_of(lam, g2) - 1) / (nd2 - 1)
    return 1.0 / phi


def main():
    print("=" * 80)
    print("CHROMATIC CORRECTION / ACHROMAT (real Sellmeier dispersion) — un-splitting colours at the focus")
    print("=" * 80)
    f_d = 100.0
    (nd1, V1), (nd2, V2) = glass_nV("N-BK7"), glass_nV("F2")
    print(f"\n  glasses (from Sellmeier): N-BK7 n_d={nd1:.4f} V_d={V1:.1f}   F2 n_d={nd2:.4f} V_d={V2:.1f}")

    fF_s, fD_s, fC_s = f_singlet(LF, "N-BK7", f_d), f_singlet(LD, "N-BK7", f_d), f_singlet(LC, "N-BK7", f_d)
    dchr_s = (fF_s - fC_s) / fD_s
    print(f"\n  SINGLET (N-BK7): f_F={fF_s:.2f} f_d={fD_s:.2f} f_C={fC_s:.2f}  chromatic shift (f_F−f_C)/f_d = {dchr_s*100:.2f}% (≈1/V={1/V1*100:.2f}%)")

    phi = 1.0 / f_d
    p1 = phi * V1 / (V1 - V2); p2 = phi * (-V2) / (V1 - V2)     # achromat: φ1+φ2=φ, φ1/V1+φ2/V2=0
    fF_a, fD_a, fC_a = (f_doublet(L, "N-BK7", "F2", p1, p2) for L in (LF, LD, LC))
    dchr_a = (fF_a - fC_a) / fD_a
    sec = (fD_a - 0.5 * (fF_a + fC_a)) / fD_a
    print(f"\n  GENERATED achromat (N-BK7 + F2): φ1={p1:.5f}(+) φ2={p2:.5f}(−), φ1/V1+φ2/V2={p1/V1+p2/V2:.1e}")
    print(f"    f_F={fF_a:.4f} f_d={fD_a:.4f} f_C={fC_a:.4f}")
    # F=C is satisfied BY CONSTRUCTION ((n_F−n_C)/(n_d−1)≡1/V by the Abbe def + the achromat solve) → tautological.
    # The GENUINE, non-tautological result is the secondary spectrum, cross-checked vs the textbook (P1−P2)/(V1−V2):
    _partial = lambda nm: (n_of(LF, nm) - n_of(LD, nm)) / (n_of(LF, nm) - n_of(LC, nm))   # relative partial dispersion
    SS_pred = (_partial("N-BK7") - _partial("F2")) / (V1 - V2)
    sec_match = abs(sec - SS_pred) / (abs(SS_pred) + 1e-30)
    print(f"    F=C common focus: shift {dchr_a*100:.4f}% — satisfied BY CONSTRUCTION (algebraic), not a measurement (was an overclaim).")
    print(f"    ★SECONDARY spectrum = {sec*100:.4f}% vs textbook (P1−P2)/(V1−V2) = {SS_pred*100:.4f}%  Δ={sec_match*100:.1f}%  (the GENUINE cross-check)")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        lams = np.linspace(0.45, 0.70, 60)
        fs = np.array([f_singlet(l, "N-BK7", f_d) for l in lams]) - f_d
        fa = np.array([f_doublet(l, "N-BK7", "F2", p1, p2) for l in lams]) - f_d
        fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=115)
        ax[0].plot(lams*1000, fs, color="#cc5555"); ax[0].axhline(0, color="k", lw=0.5)
        ax[0].set_title(f"singlet: {dchr_s*100:.1f}% chromatic shift", fontsize=9)
        ax[0].set_xlabel("λ (nm)"); ax[0].set_ylabel("focal shift Δf")
        ax[1].plot(lams*1000, fa, color="#5599ff"); ax[1].axhline(0, color="k", lw=0.5)
        for L, c in ((LF, "b"), (LC, "r")):
            ax[1].axvline(L*1000, color=c, lw=0.6, ls="--", alpha=0.5)
        ax[1].set_title(f"achromat: F=C common focus, {abs(sec)*100:.3f}% secondary spectrum", fontsize=9)
        ax[1].set_xlabel("λ (nm)")
        fig.suptitle("Achromat un-splits the colours (note the characteristic achromat 'loop')", fontsize=10)
        fig.tight_layout(); os.makedirs(_ART, exist_ok=True); fig.savefig(os.path.join(_ART, "optics_achromat.png")); plt.close(fig); rend = True
    except Exception as e:
        print(f"  (render skipped: {e})")

    # gate on the GENUINE physics: singlet IS chromatic AND the secondary spectrum MATCHES the partial-dispersion
    # formula (independent cross-check) — NOT the tautological F=C-by-construction shift (dropped from the gate).
    ok = abs(dchr_s) > 0.01 and sec_match < 0.05 and 1e-5 < abs(sec) < 5e-3
    print("\n" + "=" * 80)
    if ok:
        print("ACHROMAT validated (real Sellmeier) — chromatic correction, gated on the GENUINE cross-check:")
        print(f"  • singlet splits the focus {dchr_s*100:.2f}% (blue short, chromatic aberration).")
        print(f"  • the crown+flint achromat (φ₁/V₁+φ₂/V₂=0) brings F & C to a common focus BY CONSTRUCTION (algebraic).")
        print(f"  • ★GENUINE result: the {abs(sec)*100:.3f}% secondary spectrum MATCHES the textbook (P1−P2)/(V1−V2)=")
        print(f"    {SS_pred*100:.3f}% to {sec_match*100:.0f}% — an INDEPENDENT cross-check (not the tautological F=C shift).")
        print(f"    (Symmetric QC / the adversarial audit caught my own overclaim: the ×13e12 ratio was vacuous.)")
    else:
        print(f"  singlet {dchr_s*100:.2f}%, achromat {dchr_a*100:.4f}%, secondary {sec*100:.4f}%. Report honestly; fix at source.")
    print("=" * 80)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
