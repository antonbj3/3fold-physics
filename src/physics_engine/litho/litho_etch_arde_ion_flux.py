"""Aspect-ratio-dependent etching (ARDE / RIE lag): the ion and neutral flux transport into a high-aspect-ratio
trench that sets the etch rate. Nothing is fitted.

PHYSICS (the ARDE law emerges from the trench geometry restricting the flux; nothing prescribes the decay):
  * NEUTRALS arrive Lambertian (cosine) at the trench mouth. From the bottom centre the mouth subtends a half-angle
    alpha with tan alpha = (w/2)/d; a 2D Lambertian delivers a fraction sin alpha of its flux within +/-alpha, so
    Phi_neutral(AR) = 1/sqrt(1 + 4 AR^2) with AR = d/w, tending to 1/(2 AR) for deep trenches - the neutral
    conductance / solid-angle limit (a broad angular distribution gives strong ARDE).
  * IONS arrive near-vertical with a narrow ion angular distribution (standard deviation sigma_ion of a few degrees).
    An ion reaches the bottom only if its lateral travel over the depth stays inside the trench:
    |theta| < arctan(1/(2 AR)), so Phi_ion(AR) = erf(theta_max/(sigma_ion sqrt(2))) - ions hold flux to much higher
    aspect ratio than neutrals, which is why high-aspect-ratio etching is ion-driven.

CHECKS: (1) the etch rate (neutral flux) falls monotonically with aspect ratio (the ARDE / RIE-lag law); (2)
Phi_neutral approaches the 1/(2 AR) asymptote (the solid-angle mechanism); (3) ions maintain flux to higher aspect
ratio than neutrals; (4) for a concrete high-aspect-ratio trench (AR = 10) the neutral flux collapses to ~5% while a
sigma_ion = 3 degree ion beam still reaches the bottom.

INPUT: none. OUTPUT: printed gate lines; exit 0 when all gates hold.
"""
import sys
import numpy as np
from scipy.special import erf


def phi_neutral(AR):
    """Lambertian neutral flux fraction reaching the trench bottom center (solid-angle / view-factor limited)."""
    return 1.0 / np.sqrt(1.0 + 4.0 * AR ** 2)


def phi_ion(AR, sigma_ion_deg):
    """narrow-IAD ion flux fraction reaching the bottom (ions within θ_max=arctan(1/2AR) of vertical clear the sidewalls)."""
    theta_max = np.arctan(1.0 / (2.0 * AR))
    sig = np.radians(sigma_ion_deg)
    return erf(theta_max / (sig * np.sqrt(2.0)))


def main():
    print("=" * 104)
    print("ARDE (RIE-lag): ion+neutral flux into a HAR trench sets the etch-rate; etch-rate ↓ monotone with aspect ratio, no fit")
    print("=" * 104)
    ARs = np.array([0.5, 1, 2, 4, 6, 8, 10, 15, 20], dtype=float)
    sigma_ion = 3.0
    print(f"\n  flux to the trench bottom vs aspect ratio AR=depth/width (neutral: Lambertian; ion: narrow IAD σ={sigma_ion}°):")
    print(f"    {'AR':>5} {'Φ_neutral':>10} {'1/(2AR)':>9} {'Φ_ion':>8}")
    pn, pi = [], []
    for AR in ARs:
        n, i = phi_neutral(AR), phi_ion(AR, sigma_ion)
        pn.append(n); pi.append(i)
        print(f"    {AR:5.0f} {n:10.4f} {1/(2*AR):9.4f} {i:8.4f}")
    pn, pi = np.array(pn), np.array(pi)

    # (1) monotone decreasing (ARDE / RIE-lag)
    mono_n = all(pn[k] > pn[k + 1] for k in range(len(pn) - 1))
    mono_i = all(pi[k] >= pi[k + 1] - 1e-9 for k in range(len(pi) - 1))
    # (2) neutral → 1/(2AR) asymptote at high AR
    hi = ARs >= 8
    asym_err = float(np.max(np.abs(pn[hi] - 1.0 / (2 * ARs[hi])) / pn[hi]))
    # (3) ions hold flux to higher AR: the AR at which flux drops to 0.2
    def ar_at(frac, fn):
        g = np.linspace(0.5, 40, 8000)
        v = fn(g)
        below = np.where(v < frac)[0]
        return g[below[0]] if below.size else g[-1]
    ar_n20 = ar_at(0.2, lambda a: phi_neutral(a))
    ar_i20 = ar_at(0.2, lambda a: phi_ion(a, sigma_ion))
    # (4) concrete HAR AR=10
    n10, i10 = phi_neutral(10.0), phi_ion(10.0, sigma_ion)

    print(f"\n  ★ARDE law: BOTH fluxes decrease monotonically with AR (RIE-lag). Neutral → 1/(2AR) (solid-angle limit); ion holds via the narrow IAD.")
    print(f"  ★ions reach flux=0.2 only at AR≈{ar_i20:.1f} vs neutrals at AR≈{ar_n20:.1f} — ion-driven etch has MUCH weaker ARDE (why HAR etch is ion-driven)")
    print(f"  ★HAR AR=10: Φ_neutral={n10:.3f} (collapsed to {n10*100:.0f}%), Φ_ion={i10:.3f} ({i10*100:.0f}%) — the neutral-starved, ion-sustained HAR regime")

    g1 = mono_n and mono_i                                          # ★etch-rate ↓ monotone with AR
    g2 = asym_err < 0.02                                            # ★neutral → 1/(2AR) solid-angle asymptote
    g3 = ar_i20 > 3.0 * ar_n20                                      # ★ions hold flux to ≫ higher AR than neutrals
    g4 = n10 < 0.10 and i10 > 0.3                                   # ★HAR AR=10: neutral collapsed, ion sustained
    ok = g1 and g2 and g3 and g4
    print("\n" + "-" * 104)
    print(f"  (1) ★etch-rate ↓ MONOTONE with aspect ratio (ARDE/RIE-lag), both neutral & ion   {'OK' if g1 else 'FAIL'}")
    print(f"  (2) ★neutral flux → 1/(2AR) asymptote (max rel {asym_err*100:.1f}% for AR≥8) — the solid-angle/conductance mechanism   {'OK' if g2 else 'FAIL'}")
    print(f"  (3) ★ions hold flux to AR≈{ar_i20:.0f} vs neutrals AR≈{ar_n20:.1f} (>{ar_i20/ar_n20:.0f}×) — narrow IAD ⇒ weak ARDE (ion-driven HAR etch)   {'OK' if g3 else 'FAIL'}")
    print(f"  (4) ★HAR AR=10: Φ_neutral={n10:.3f}<0.10 (collapsed), Φ_ion={i10:.3f}>0.3 (sustained) — the physical HAR-etch regime   {'OK' if g4 else 'FAIL'}")
    print("=" * 104)
    if ok:
        print("The ARDE etch-rate law (ion+neutral flux into a high-aspect-ratio trench) certified, no fit:")
        print(f"  • ★the RIE-LAG / ARDE law EMERGES from geometry: a Lambertian neutral flux to the bottom is solid-angle-limited to Φ=1/√(1+4AR²)")
        print(f"    → 1/(2AR) (the mouth shrinks in the bottom's view as the trench deepens); the etch-rate ↓ monotone with AR, from first principles.")
        print(f"  • ★the ION/NEUTRAL SPLIT is the mechanism: a narrow ion IAD (σ={sigma_ion}°) clears the sidewalls to AR≈{ar_i20:.0f} while neutrals starve by AR≈{ar_n20:.1f}")
        print(f"    — high-aspect-ratio etching is ion-flux-driven, which is why the front velocity is prescribed from the ion flux. At AR=10 neutrals are at {n10*100:.0f}%, ions {i10*100:.0f}%.")
        print(f"  • ★this is the flux model that sets the local front velocity v(AR); the level-set front advance (v → geometry) is done in")
        print(f"    litho_etch_levelset_ballistic.py. 0 free parameters.")
    else:
        print(f"  HONEST: mono_n={mono_n} mono_i={mono_i} asym={asym_err:.3f} ar_i20={ar_i20:.1f} ar_n20={ar_n20:.1f} n10={n10:.3f} i10={i10:.3f}. Inspect.")
    print("=" * 104)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
