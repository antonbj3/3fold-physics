"""ANTENNA GAIN sigma-BUDGET — RENDER->MATCH what limits how precisely you know a dish's gain, and the lesson that SENSITIVITY
alone does not set a budget. Importing gain_dBi(D,f,eta), the gain in dB is 20log10(D)+20log10(f)+10log10(eta)+const, so the
diameter and frequency enter with TWICE the per-fractional sensitivity of the efficiency (they are squared in linear gain). Yet
the gain uncertainty is dominated by the EFFICIENCY -- because a dish's diameter and frequency are known to ~1% while its
aperture efficiency is a ~10% guess. Contribution = (sensitivity x input-vagueness)^2; the vaguer input wins even at half the
sensitivity. Uses render_match_scaffold.

MATCH: a 1 m / 10 GHz / eta=0.6 dish (sigma 1cm, 0.5%, 10%) has a gain uncertainty ~0.45 dB (MC=Jacobian); efficiency is ~95%
of it though it is the LEAST sensitive input. render->match, never fit.

  python em/antenna_sigma_budget.py
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
from antenna_aperture import gain_dBi                             # gain_dBi(D,f,eta) in dBi

D0, F0, ETA0, S_D, S_F = 1.0, 10e9, 0.60, 0.01, 5e7              # m, Hz, -, sigma_D [m], sigma_f [Hz]


def budget(s_eta):
    dD = (gain_dBi(D0 + 1e-4, F0, ETA0) - gain_dBi(D0 - 1e-4, F0, ETA0)) / 2e-4
    dF = (gain_dBi(D0, F0 + 1e4, ETA0) - gain_dBi(D0, F0 - 1e4, ETA0)) / 2e4
    dE = (gain_dBi(D0, F0, ETA0 + 1e-5) - gain_dBi(D0, F0, ETA0 - 1e-5)) / 2e-5
    c = np.array([(dD * S_D) ** 2, (dF * S_F) ** 2, (dE * s_eta) ** 2])   # D, f, eta
    return np.sqrt(c.sum()), c / c.sum(), (dD * D0, dE * ETA0)    # sigma_G, [fD,ff,fEta], per-relative dB sens (D, eta)


def mc_sigma(s_eta, n=400000, seed=3):
    g = np.random.default_rng(seed)
    return gain_dBi(g.normal(D0, S_D, n), g.normal(F0, S_F, n), g.normal(ETA0, s_eta, n)).std()


def main():
    print("=" * 92)
    print("ANTENNA GAIN sigma-BUDGET — D,f squared but efficiency (vaguest) dominates; sensitivity x uncertainty")
    print("=" * 92)
    sG_mc = mc_sigma(0.06)
    def rfn(p):
        return budget(p["se"])[0]
    band = [{"se": 0.04}, {"se": 0.08}]                           # how poorly we know aperture efficiency = the band
    res = render_match(
        rfn, band, {"se": 0.06},
        Benchmark("gain uncertainty sigma_G (MC), 1m/10GHz dish", round(sG_mc, 3), 0.06, "Monte Carlo 400k", "dB"),
        nulls=[("calibrate the efficiency (sigma_eta->0 -> spread drops sharply)", {"se": 1e-6}, lambda s, m: s < budget(0.06)[0] / 1.5)],
        perturbations=[("efficiency a wilder guess (sigma_eta up -> more spread)", {"se": 0.12}, lambda s, best: s > best)],
        notes=["efficiency dominates, so pinning it collapses the spread; a vaguer efficiency widens it"])
    print(res.report())
    s_std, f_std, (srel_D, srel_E) = budget(0.06)
    print(f"\n  budget [D, f, eta] = {np.round(f_std*100,0)}%   per-fractional dB sensitivity: D={srel_D:.1f}, eta={srel_E:.1f} dB")
    print(f"  (4) ★EFFICIENCY DOMINATES ({f_std[2]*100:.0f}%) THOUGH LEAST SENSITIVE: D,f enter squared so each is {srel_D/srel_E:.0f}x more sensitive per fraction than eta — yet eta wins on a 10% vagueness vs the dish's 1%")
    print(f"  (5) ★SENSITIVITY x UNCERTAINTY: a contribution is (sensitivity*sigma)^2, not sensitivity alone — measure the dish to 1mm all you like, the gain stays {s_std:.2f} dB uncertain until you calibrate the efficiency")
    g4 = f_std[2] > 0.85 and abs(s_std - sG_mc) / sG_mc < 0.05    # eta dominates; Jacobian≈MC
    g5 = abs(srel_D / srel_E - 2.0) < 0.05 and f_std[0] < f_std[2]   # D 2x more sensitive yet contributes less
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (antenna sigma-budget) — sensitivity x uncertainty, Jacobian vs MC:")
        print(f"  • the MC gain uncertainty {res.best:.2f} dB (band [{res.band_lo:.2f},{res.band_hi:.2f}]=efficiency-knowledge σ) matches the Jacobian to <5%.")
        print(f"  • the efficiency is {f_std[2]*100:.0f}% of it though D,f are {srel_D/srel_E:.0f}x more sensitive — a squared-but-precise input loses to a linear-but-vague one.")
        print(f"  • the budget is set by sensitivity x input-uncertainty, not the exponent; to know the gain, calibrate the efficiency, not the ruler.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, eta-dominates/Jac≈MC {g4}, sens-flip {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
