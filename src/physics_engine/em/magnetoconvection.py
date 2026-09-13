"""MAGNETOCONVECTION (Chandrasekhar) — a vertical MAGNETIC FIELD SUPPRESSES thermal convection. This is a genuine
THREE-FIELD coupling (thermal⊗flow⊗em): buoyancy drives Rayleigh-Bénard convection, but the Lorentz force on the induced
currents resists the overturning, so the onset is DELAYED. The linear marginal-stability analysis of the magneto-Boussinesq
equations (free-free boundaries) gives the neutral Rayleigh number as a function of horizontal wavenumber a and the
Chandrasekhar number Q = B²d²σ/(ρνμ_mag-ish) (the magnetic-to-viscous stress ratio):

    Ra(a², Q) = [ (π²+a²)³ + π² Q (π²+a²) ] / a²

The critical Ra_c(Q) and the cell size a_c(Q) are NOT coded — they fall out of MINIMISING this over a². ★The magnetic term
raises Ra_c (stabilisation) AND pushes the minimiser to larger a_c (the convection cells get NARROWER under the field — a
non-obvious, physical prediction: the field channels motion into thin sheets aligned with B).

★Anchors (Ra_c, a_c come from the minimisation; only the dispersion relation + a² grid are coded):
  • NULL Q=0 recovers ordinary Rayleigh-Bénard: Ra_c = 27π⁴/4 ≈ 657.5 at a_c = π/√2 — emergent;
  • ★Ra_c INCREASES monotonically with Q (magnetic stabilisation), → asymptotically LINEAR in Q (strong-field π²Q regime);
  • ★the critical wavenumber a_c INCREASES with Q (cells narrow) — the field constrains horizontal overturning.

  python em/magnetoconvection.py
"""
import sys
import numpy as np
from scipy.optimize import minimize_scalar


def Ra_marginal(a2, Q):
    return ((np.pi ** 2 + a2) ** 3 + np.pi ** 2 * Q * (np.pi ** 2 + a2)) / a2


def critical(Q):
    """minimise the neutral Ra over a² ⇒ (Ra_c, a_c) for a given Chandrasekhar number Q (robust across all Q)."""
    r = minimize_scalar(lambda la2: Ra_marginal(np.exp(la2), Q),
                        bounds=(np.log(1e-3), np.log(1e5)), method="bounded")
    a2 = np.exp(r.x)
    return float(Ra_marginal(a2, Q)), float(np.sqrt(a2))


def main():
    print("=" * 90)
    print("MAGNETOCONVECTION — a magnetic field raises the convection threshold Ra_c(Q) (thermal⊗flow⊗em)")
    print("=" * 90)

    Qs = [0.0, 1.0, 10.0, 100.0, 1000.0, 1e4, 1e5, 1e6]
    print(f"\n  Q (Chandrasekhar):   Ra_c          a_c       Ra_c/Q")
    Rac, ac = [], []
    for Q in Qs:
        R, a = critical(Q); Rac.append(R); ac.append(a)
        print(f"    Q={Q:<8g}   Ra_c = {R:11.2f}   a_c = {a:.3f}   Ra_c/Q = {R/Q if Q else float('nan'):.3f}")
    Rac = np.array(Rac); ac = np.array(ac)

    # NULL Q=0: the classic Rayleigh-Bénard free-free result
    Ra0_an = 27 * np.pi ** 4 / 4; ac0_an = np.pi / np.sqrt(2)
    null_ok = abs(Rac[0] - Ra0_an) / Ra0_an < 1e-3 and abs(ac[0] - ac0_an) / ac0_an < 1e-2

    # ★stabilisation: Ra_c monotone increasing in Q
    mono_Ra = all(Rac[i] < Rac[i + 1] for i in range(len(Rac) - 1))
    # ★asymptotically LINEAR in Q at strong field: Ra_c(Q)/Q → π² (Chandrasekhar). Check the top decade.
    asym_ratio = Rac[-1] / Qs[-1]                                   # Ra_c/Q at Q=1e6
    asym_linear = abs(asym_ratio - np.pi ** 2) / np.pi ** 2 < 0.05 and abs(Rac[-1] / Rac[-2] - 10) < 0.5
    # ★cell narrowing: a_c monotone increasing in Q
    mono_a = all(ac[i] < ac[i + 1] for i in range(len(ac) - 1))

    g1 = null_ok
    g2 = mono_Ra and asym_linear                                   # rises with Q, → linear (Ra_c ≈ π²Q) at strong field
    g3 = mono_a and ac[-1] > 1.5 * ac[0]                           # cells clearly narrower at high Q
    ok = g1 and g2 and g3
    print(f"\n  (1) NULL Q=0 ⇒ Rayleigh-Bénard: Ra_c={Rac[0]:.2f} vs 27π⁴/4={Ra0_an:.2f}, a_c={ac[0]:.3f} vs π/√2={ac0_an:.3f}  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★STABILISATION → LINEAR in Q: Ra_c monotone↑ ({Rac[0]:.0f}→{Rac[-1]:.0f}); Ra_c/Q → {asym_ratio:.3f} vs π²={np.pi**2:.3f} (strong-field limit)  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) ★CELLS NARROW: a_c monotone↑ in Q, {ac[0]:.3f}→{ac[-1]:.3f} ({ac[-1]/ac[0]:.1f}× — thinner cells under B)  {'✓' if g3 else 'FAIL'}")
    print("\n" + "=" * 90)
    if ok:
        print("VALIDATED: magnetoconvection — a magnetic field suppresses convection (thermal⊗flow⊗em, non-tautological):")
        print(f"  • NULL Q=0 recovers ordinary Rayleigh-Bénard exactly (Ra_c={Rac[0]:.1f} = 27π⁴/4, a_c=π/√2) — emergent from min over a².")
        print(f"  • ★Ra_c rises monotonically with the Chandrasekhar number Q ({Rac[0]:.0f}→{Rac[-1]:.0f}, {Rac[-1]/Rac[0]:.0f}× by Q=1000), → linear")
        print(f"    in Q at strong field — the Lorentz force on induced currents resists overturning, delaying onset.")
        print(f"  • ★the critical cell wavenumber a_c GROWS with Q ({ac[0]:.2f}→{ac[-1]:.2f}) — convection is forced into NARROW sheets")
        print(f"    aligned with B. A three-field coupling (sunspots, the solar tachocline, liquid-metal blankets, MHD heat transfer).")
    else:
        print(f"  (1)null {g1} (2)stabilise {g2} ({Rac[-1]/Rac[0]:.0f}×) (3)narrow {g3} ({ac[-1]/ac[0]:.1f}×). Fix at source.")
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
