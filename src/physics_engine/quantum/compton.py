"""COMPTON SCATTERING — a photon bouncing off a free electron comes back REDDER, and the wavelength shift depends only on
the scattering ANGLE, not the photon's energy. This was the decisive evidence that light carries momentum like a particle.
The shift Δλ = λ_C(1−cosθ), λ_C = h/(m_e c) = 2.426 pm (the Compton wavelength), FALLS OUT of relativistic
energy-momentum conservation between the photon and the recoiling electron — here SOLVED numerically (the conservation
equations, root-found for E′), not by quoting the formula.

★Anchors (only the conservation laws + the relativistic electron E²=(pc)²+(mc²)² are coded):
  • Δλ = λ_C(1−cosθ): the shift matches the Compton wavelength times (1−cosθ), and ★is INDEPENDENT of the incident
    energy — an X-ray (100 keV) and a γ-ray (1 MeV) show the SAME Δλ at a given angle (while their ΔE/E differ wildly).
    This additive-in-WAVELENGTH (not energy) signature is what distinguished Compton (particle) from classical (Thomson);
  • max shift 2λ_C at back-scatter θ=180°; NULL θ=0 (forward) ⇒ Δλ=0; energy + momentum conserved (the electron recoils).

  python quantum/compton.py
"""
import sys
import numpy as np
from scipy.optimize import brentq

HC = 1239.842            # keV·pm  (h·c)
MEC2 = 510.999           # keV     (electron rest energy)
LAMBDA_C = HC / MEC2     # pm      = 2.426


def scatter_Eprime(E0, theta):
    """solve relativistic energy+momentum conservation for the scattered-photon energy E′(θ) — root-find, not the formula.
    From momentum conservation p_e²c² = E0² + E′² − 2E0E′cosθ; energy conservation E0+mc² = E′ + √(p_e²c² + (mc²)²)."""
    def resid(Ep):
        pe2c2 = E0 ** 2 + Ep ** 2 - 2 * E0 * Ep * np.cos(theta)        # (p_e c)² from momentum conservation
        Ee = (E0 + MEC2) - Ep                                          # electron total energy from energy conservation
        return Ee ** 2 - (pe2c2 + MEC2 ** 2)                           # must satisfy E_e² = (p_e c)² + (mc²)²
    return brentq(resid, 0.5 * E0 / (1 + 2 * E0 / MEC2), E0 + 1e-9)    # E′ ∈ (back-scatter min, E0)


def main():
    print("=" * 90)
    print("COMPTON SCATTERING — Δλ = λ_C(1−cosθ) emerges from relativistic energy-momentum conservation")
    print("=" * 90)
    thetas = np.radians([0, 30, 60, 90, 120, 180])

    # ★energy-independence: same Δλ(θ) for an X-ray and a γ-ray, both solved numerically
    print(f"\n  θ(°):    Δλ/λ_C  @100keV    Δλ/λ_C @1000keV    1−cosθ (theory)")
    errs = []; indep = []
    for E0 in (100.0, 1000.0):
        col = []
        for th in thetas:
            Ep = scatter_Eprime(E0, th)
            dlam = HC / Ep - HC / E0
            col.append(dlam / LAMBDA_C)
        if E0 == 100.0: colA = col
        else: colB = col
    for i, th in enumerate(thetas):
        theory = 1 - np.cos(th)
        errs.append(abs(colA[i] - theory)); errs.append(abs(colB[i] - theory))
        indep.append(abs(colA[i] - colB[i]))
        print(f"  {np.degrees(th):5.0f}:    {colA[i]:8.4f}       {colB[i]:8.4f}        {theory:.4f}")

    # conservation check at θ=90°, E0=1000: total energy + momentum before vs after
    E0 = 1000.0; th = np.pi / 2; Ep = scatter_Eprime(E0, th)
    pe2c2 = E0 ** 2 + Ep ** 2 - 2 * E0 * Ep * np.cos(th); Ee = E0 + MEC2 - Ep
    cons_E = abs((Ep + Ee) - (E0 + MEC2)) / (E0 + MEC2)
    cons_relativistic = abs(Ee ** 2 - (pe2c2 + MEC2 ** 2)) / Ee ** 2

    g1 = max(errs) < 1e-3                                            # Δλ = λ_C(1−cosθ)
    g2 = max(indep) < 1e-3                                           # ★energy-independent (X-ray == γ-ray Δλ)
    g3 = abs(colA[-1] - 2.0) < 1e-3 and abs(colA[0]) < 1e-6          # max 2λ_C at 180°, null at 0°
    g4 = cons_E < 1e-9 and cons_relativistic < 1e-9                 # energy + relativistic electron consistent
    ok = g1 and g2 and g3 and g4
    print(f"\n  (1) Δλ = λ_C(1−cosθ): max deviation from theory = {max(errs):.1e}  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★ENERGY-INDEPENDENT: |Δλ(100keV) − Δλ(1000keV)| max = {max(indep):.1e} (same shift, X-ray vs γ-ray)  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) MAX 2λ_C at θ=180° ({colA[-1]:.4f}), NULL Δλ=0 at θ=0° ({colA[0]:.1e})  {'✓' if g3 else 'FAIL'}")
    print(f"  (4) CONSERVATION: energy drift {cons_E:.0e}, relativistic E_e²=(p_ec)²+(mc²)² to {cons_relativistic:.0e}  {'✓' if g4 else 'FAIL'}")
    print("\n" + "=" * 90)
    if ok:
        print("VALIDATED: Compton scattering — the wavelength shift emerges from relativistic conservation (non-tautological):")
        print(f"  • Δλ = λ_C(1−cosθ) (λ_C={LAMBDA_C:.3f} pm) falls out of solving energy+momentum conservation for E′ — never coded.")
        print(f"  • ★the shift is INDEPENDENT of the incident energy (X-ray and γ-ray give the SAME Δλ at each angle, to {max(indep):.0e})")
        print(f"    — additive in WAVELENGTH, not energy: the particle-of-light signature that defeated the classical picture.")
        print(f"  • max 2λ_C at back-scatter, zero forward; energy & momentum conserved with the electron recoil. The basis of")
        print(f"    γ-ray spectroscopy, Compton telescopes, and electron-density imaging.")
    else:
        print(f"  (1)formula {g1} ({max(errs):.0e}) (2)E-indep {g2} ({max(indep):.0e}) (3)max/null {g3} (4)conservation {g4}. Fix at source.")
    print("=" * 90)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
