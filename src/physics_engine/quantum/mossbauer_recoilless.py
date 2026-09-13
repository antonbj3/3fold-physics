r"""MÖSSBAUER EFFECT — THE RECOILLESS FRACTION — RENDER->MATCH why a nucleus bound in a crystal can emit and absorb a
gamma ray with NO recoil, giving an absurdly sharp resonance (the basis of the most precise spectroscopy ever, sensitive to
1 part in 10^15). A free nucleus emitting a gamma of energy E recoils with E_R = E^2/(2 M c^2), shifting the line off resonance by
far more than its natural width -> no resonant re-absorption. But in a LATTICE, a fraction f of emissions transfer the recoil to
the WHOLE crystal (effectively infinite mass), so the gamma carries the full energy and the line stays needle-sharp. That
recoilless fraction is the Debye-Waller factor
    f = exp(-k^2 <x^2>) = exp(-2W),    2W = (E_R / k_B Theta_D) [3/2 + 6 (T/Theta_D)^2 \int_0^{Theta_D/T} y/(e^y-1) dy],
the same <x^2> that smears X-ray Bragg peaks. We compute f from the GENUINE Debye phonon mean-square displacement (the integral,
not a limit), for 57Fe (14.4 keV).

MATCH: the 57Fe recoilless fraction at room temperature, from the Debye-Waller <x^2>, sits at ~0.7-0.8 (measured in iron); it falls monotonically with temperature (more thermal vibration), it survives even at T=0 (zero-point motion keeps f<1), it vanishes for an unbound atom (no lattice -> full recoil), and a higher-energy gamma recoils more and lowers it.
  python quantum/mossbauer_recoilless.py
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
from scipy.integrate import quad
from render_match_scaffold import Benchmark, render_match

KB = 8.617333e-5                                          # eV/K
FE57_MEAS = 0.78                                          # 57Fe recoilless fraction in iron at 300 K (~0.7-0.8 measured)


def recoilless_f(Egamma=14.4e3, A=57, ThetaD=420.0, T=300.0):
    """Debye-Waller recoilless fraction f = exp(-2W) from the GENUINE Debye phonon <x^2> integral. ThetaD->0 = unbound (no lattice)."""
    if ThetaD < 1.0:
        return 0.0                                        # no lattice binding -> full recoil, no recoilless line
    ER = Egamma ** 2 / (2 * A * 931.494e6)                # recoil energy (eV); M c^2 = A * 931.494 MeV
    integ = quad(lambda y: y / np.expm1(y), 1e-9, ThetaD / T)[0]
    twoW = (ER / (KB * ThetaD)) * (1.5 + 6 * (T / ThetaD) ** 2 * integ)
    return float(np.exp(-twoW))


def main():
    print("=" * 96)
    print("MÖSSBAUER RECOILLESS FRACTION — f=exp(-2W) from the Debye-Waller <x^2>; render->match")
    print("=" * 96)
    def rfn(p):
        return recoilless_f(Egamma=p.get("Egamma", 14.4e3), A=p.get("A", 57), ThetaD=p.get("ThetaD", 420.0), T=p.get("T", 300.0))
    band = [{"ThetaD": 380.0}, {"ThetaD": 470.0}]         # lattice (Debye-temperature) σ: 57Fe Theta_D ~380-470K across hosts
    res = render_match(
        rfn, band, {"ThetaD": 420.0},
        Benchmark("57Fe recoilless fraction f (300K)", FE57_MEAS, 0.05, "measured 57Fe Mössbauer fraction in iron ~0.7-0.8 (EXTERNAL)", ""),
        nulls=[("an UNBOUND atom (no lattice, Theta_D->0) takes the full recoil -- the line shifts off resonance, no recoilless emission (free -> f->0)", {"ThetaD": 0.5}, lambda v, m: v < m / 5)],
        perturbations=[("a hotter lattice vibrates more, so fewer emissions are recoilless (T up -> lower f)", {"T": 600.0}, lambda v, best: v < best - 0.05)],
        notes=["the 57Fe recoilless fraction from the Debye-Waller <x^2> matches the measured ~0.78; it falls with temperature; an unbound atom has f=0"])
    print(res.report())
    # ★the value + the temperature trend + the zero-point survival + the recoil-vs-width
    f300 = recoilless_f()
    ER = 14.4e3 ** 2 / (2 * 57 * 931.494e6)
    print(f"\n  57Fe (14.4 keV): recoil energy E_R = {ER*1e3:.3f} meV; recoilless fraction f(300K) = {f300:.3f}  vs  measured ~{FE57_MEAS}")
    print(f"  ★RECOIL vs LINEWIDTH: E_R={ER*1e3:.2f} meV is ~{ER/4.7e-9:.0e}x the natural linewidth (4.7e-9 eV) -- a FREE 57Fe line recoils completely off resonance; only the recoilless fraction f stays sharp enough to resonate")
    print(f"  ★FALLS WITH TEMPERATURE: f(T) = " + ", ".join(f"{T}K:{recoilless_f(T=T):.3f}" for T in (4, 80, 300, 600)) + " -- more thermal phonons = larger <x^2> = fewer recoilless events (why Mössbauer works best cold)")
    print(f"  ★ZERO-POINT SURVIVES T=0: f(T->0)={recoilless_f(T=1):.3f} < 1 -- even at absolute zero the quantum zero-point motion (the 3/2 term, NOT thermal) keeps <x^2>>0, so f never reaches 1; a purely classical lattice would give f=1 at T=0")
    print(f"  ★HEAVIER/HIGHER-E IS HARDER: E_R ~ E_gamma^2, so f = " + ", ".join(f"{n}:{recoilless_f(Egamma=eg,A=a,ThetaD=td):.2f}" for n,eg,a,td in [('57Fe-14keV',14.4e3,57,420),('119Sn-24keV',23.9e3,119,200),('191Ir-129keV',129e3,191,285)]) + " -- the 129 keV Ir line is essentially un-observable recoilless (E_R too big); low-energy transitions in stiff lattices are ideal")
    print(f"  (4) ★f FROM THE PHONON <x^2> (not assumed): integrating the Debye spectrum for <x^2> and forming exp(-k^2<x^2>) gives f(300K)={f300:.3f}, in the measured 0.7-0.8 band -- the SAME <x^2> that damps X-ray Bragg peaks (Debye-Waller); the recoilless fraction IS a lattice-dynamics quantity")
    print(f"  (5) ★WHY IT MATTERS: the recoilless line is so sharp (Delta E/E ~10^-15) that it resolves the 14-keV gamma's gravitational red-shift over 22 m (Pound-Rebka, testing GR), hyperfine fields, isomer shifts and quadrupole splittings in materials/biology/geology, and it is a frequency standard -- all enabled by f being a sizeable fraction, not zero")
    g4 = abs(f300 - FE57_MEAS) < 0.06 and recoilless_f(T=600) < recoilless_f(T=80) and recoilless_f(T=1) < 1.0  # value; T-monotone; zero-point<1
    g5 = recoilless_f(ThetaD=0.5) < 0.05 and recoilless_f(Egamma=129e3, A=191, ThetaD=285) < f300  # free-atom null; higher-E harder
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Mössbauer recoilless fraction) — the lattice-bound nucleus that doesn't recoil:")
        print(f"  • 57Fe f(300K)={res.best:.3f} matches the measured ~0.78 (band [{res.band_lo:.3f},{res.band_hi:.3f}]=Debye-temperature σ).")
        print(f"  • from the Debye-Waller <x^2>; falls with T; survives (f<1) at T=0 by zero-point motion; f=0 for an unbound atom.")
        print(f"  • ★the recoil goes to the whole crystal, leaving a needle-sharp gamma line -- the most precise spectroscopy there is.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, value/T-monotone/zero-point {g4}, free-null/higher-E {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
