"""TWO-BAND MAGNETORESISTANCE — RENDER->MATCH why a metal's electrical resistance RISES in a magnetic field -- but
only if it carries current in MORE THAN ONE channel. A single carrier species in the Drude model has its Hall field exactly
cancel the magnetic deflection, so its longitudinal resistance is FIELD-INDEPENDENT (zero magnetoresistance, a remarkable null).
With two carriers of different mobility, the Hall field cannot balance both at once; they deflect by different amounts, paths
lengthen, and the resistance grows as B^2 at low field and SATURATES at high field. From the two-carrier conductivity tensor
    sigma_xx = sum_i n_i e mu_i/(1+(mu_i B)^2),   sigma_xy = sum_i n_i e mu_i^2 B/(1+(mu_i B)^2),
inverting to rho_xx gives a low-field magnetoresistance coefficient (Delta rho/rho)/B^2 = sigma_1 sigma_2 (mu_1-mu_2)^2/(sigma_1+sigma_2)^2.
We do NOT assume it: we build the tensor, INVERT it numerically, and read the coefficient. render_match_scaffold.

MATCH: the low-field two-band magnetoresistance coefficient, from numerically inverting the conductivity tensor, equals the analytic sigma_1 sigma_2 (mu_1-mu_2)^2/(sigma_1+sigma_2)^2; a single carrier species gives ZERO magnetoresistance; a larger mobility contrast gives more.
  python em/magnetoresistance_twoband.py
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
from render_match_scaffold import Benchmark, render_match

N1, MU1, N2, MU2 = 1.0, 1.0, 1.0, 3.0                     # two carrier bands (e=1); band 2 more mobile
E = 1.0


def rho_xx(B, mu2=MU2, n1=N1, mu1=MU1, n2=N2):
    """longitudinal resistivity from the INVERTED two-band conductivity tensor."""
    sxx = n1 * E * mu1 / (1 + (mu1 * B) ** 2) + n2 * E * mu2 / (1 + (mu2 * B) ** 2)
    sxy = n1 * E * mu1 ** 2 * B / (1 + (mu1 * B) ** 2) + n2 * E * mu2 ** 2 * B / (1 + (mu2 * B) ** 2)
    return sxx / (sxx ** 2 + sxy ** 2)


def mr_coeff(mu2=MU2, B=0.01):
    """low-field magnetoresistance coefficient (Delta rho/rho)/B^2, from the numerical tensor inversion at small B."""
    r0 = rho_xx(0.0, mu2=mu2)
    return float((rho_xx(B, mu2=mu2) - r0) / r0 / B ** 2)


def mr_analytic(mu2=MU2, n1=N1, mu1=MU1, n2=N2):
    s1, s2 = n1 * E * mu1, n2 * E * mu2
    return float(s1 * s2 * (mu1 - mu2) ** 2 / (s1 + s2) ** 2)


def main():
    print("=" * 96)
    print("TWO-BAND MAGNETORESISTANCE — low-field coeff from the inverted conductivity tensor; render->match")
    print("=" * 96)
    def rfn(p):
        return mr_coeff(mu2=p.get("mu2", MU2))
    band = [{"mu2": 0.9 * MU2}, {"mu2": 1.1 * MU2}]       # carrier-mobility σ (±10%): coeff ~ (mu1-mu2)^2
    res = render_match(
        rfn, band, {"mu2": MU2},
        Benchmark("low-field MR coeff (Drho/rho)/B^2", mr_analytic(), 0.04, "sigma1 sigma2 (mu1-mu2)^2/(sigma1+sigma2)^2 (EXTERNAL)", "1/B^2"),
        nulls=[("a SINGLE carrier species (mu2=mu1) has its Hall field exactly cancel the deflection -- ZERO magnetoresistance, the resistance is field-independent (mu2->mu1 -> coeff->0)", {"mu2": MU1}, lambda v, m: v < m / 10)],
        perturbations=[("a LARGER mobility contrast makes the two carriers deflect more differently -- more magnetoresistance (mu2 up -> larger coeff)", {"mu2": 5.0}, lambda v, best: v > best + 0.5)],
        notes=["the inverted-tensor low-field coefficient matches sigma1 sigma2 (mu1-mu2)^2/(sigma1+sigma2)^2; a single band gives zero MR; more contrast gives more"])
    print(res.report())
    # ★the coefficient + the single-band null + the B^2->saturation + the Hall cancellation
    print(f"\n  low-field coeff (Drho/rho)/B^2: numerical(tensor inversion)={mr_coeff():.4f} vs analytic={mr_analytic():.4f}")
    print(f"  ★ONE CARRIER = ZERO MR (the Drude null): a single species has (Drho/rho)/B^2 = {mr_coeff(mu2=MU1):.2e} -- its Hall field E_y exactly balances the Lorentz force, so electrons drift straight and rho_xx is field-independent. Magnetoresistance REQUIRES multiple channels (bands, anisotropy, or open orbits)")
    print(f"  ★B^2 THEN SATURATES: Delta rho/rho at B = " + ", ".join(f"{b}:{(rho_xx(b)-rho_xx(0))/rho_xx(0):.3f}" for b in (0.1, 1.0, 10.0)) + " -- quadratic at low field (the coeff), saturating at high field when omega_c tau >> 1 (the slow band dominates). The saturation value fingerprints the carrier densities")
    print(f"  ★MOBILITY CONTRAST DRIVES IT: coeff ~ (mu1-mu2)^2 -- " + ", ".join(f"mu2={m}:{mr_analytic(mu2=m):.3f}" for m in (1.0, 2.0, 3.0, 5.0)) + "; equal mobilities (even with two bands) give zero -- it is the DIFFERENCE in deflection that lengthens the paths")
    print(f"  (4) ★COEFF FROM TENSOR INVERSION (not assumed): building sigma_xx, sigma_xy for two bands and inverting to rho_xx=sigma_xx/(sigma_xx^2+sigma_xy^2) gives {mr_coeff():.4f}, matching the closed form to {abs(mr_coeff()-mr_analytic())/mr_analytic()*100:.2f}% -- the magnetoresistance is a pure consequence of the tensor structure, no extra physics")
    print(f"  (5) ★WHY IT MATTERS: two-band magnetoresistance measures carrier densities and mobilities (the Kohler/mobility-spectrum analysis), is the basis of magnetic-field sensors and read heads (giant/ordinary MR), flags multi-band superconductors and topological semimetals (extreme non-saturating MR), and the single-band NULL is the textbook proof that simple metals don't magnetoresist")
    g4 = abs(mr_coeff() - mr_analytic()) / mr_analytic() < 0.03 and mr_coeff(mu2=MU1) < 1e-6  # numerical=analytic; single-band null
    g5 = mr_analytic(mu2=5.0) > mr_analytic(mu2=3.0) > mr_analytic(mu2=2.0) and (rho_xx(10.0) - rho_xx(0)) / rho_xx(0) > (rho_xx(1.0) - rho_xx(0)) / rho_xx(0)  # contrast monotone; saturating growth
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (two-band magnetoresistance) — resistance that needs two channels to rise:")
        print(f"  • low-field coeff {res.best:.4f} matches sigma1 sigma2 (mu1-mu2)^2/(sigma1+sigma2)^2 (band [{res.band_lo:.4f},{res.band_hi:.4f}]=mobility σ).")
        print(f"  • a single carrier gives ZERO MR (Hall cancellation); B^2 then saturating; the mobility contrast is the drive.")
        print(f"  • ★magnetoresistance is a tensor-inversion consequence -- it exists only because two carriers deflect differently.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, numerical/single-null {g4}, contrast/saturate {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
