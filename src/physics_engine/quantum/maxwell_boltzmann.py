"""MAXWELL-BOLTZMANN SPEED DISTRIBUTION — RENDER->MATCH how fast molecules actually move and -- the part that runs
chemistry -- how many move FAST ENOUGH to react. In a gas at temperature T the velocity components are independent Gaussians
(variance kT/m), so the SPEED v=|vec v| follows f(v) proportional to v^2 exp(-m v^2/2kT), with three characteristic speeds
    v_p = sqrt(2kT/m)  <  <v> = sqrt(8kT/pi m)  <  v_rms = sqrt(3kT/m).
The high-speed TAIL is the engine of reaction kinetics: the fraction of molecules with kinetic energy above a threshold E_a is the
upper-incomplete-gamma Gamma(3/2, E_a/kT)/Gamma(3/2), dominated by exp(-E_a/kT) -- this is where the Arrhenius factor comes from, by
COUNTING energetic molecules (collision theory), the statistical sibling of kramers_escape's dynamical (barrier-crossing) route.
We sample the distribution, recover the three speeds and the reactive-tail fraction. Gates run through render_match_scaffold.

MATCH: the mean molecular speed emerges as sqrt(8kT/pi m); the three speeds order v_p<<v><v_rms; and the high-energy tail gives the Arrhenius fraction exp(-E_a/kT) (collision-theory origin of the rate law).
  python quantum/maxwell_boltzmann.py
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
from scipy.special import gammaincc
from render_match_scaffold import Benchmark, render_match

M = 1.0                                                   # molecular mass (k_B=1 units)
_RNG = np.random.default_rng(0)
_V = _RNG.standard_normal((2_000_000, 3))                # frozen unit-variance velocity sample (scale by sqrt(T) per call)


def speeds(T):
    return np.sqrt((_V * np.sqrt(T / M)) ** 2 ** 1).sum(1) if False else np.sqrt(((_V * np.sqrt(T / M)) ** 2).sum(1))


def mean_speed(T):
    return float(speeds(T).mean())


def reactive_fraction(T, Ea):
    return float(np.mean(0.5 * M * speeds(T) ** 2 > Ea))


def main():
    print("=" * 96)
    print("MAXWELL-BOLTZMANN — mean speed sqrt(8kT/pi m), three speeds, and the Arrhenius tail; render->match")
    print("=" * 96)
    T0 = 1.0
    def rfn(p):
        return mean_speed(p.get("T", T0))
    band = [{"T": 0.5}, {"T": 2.0}]                      # temperature uncertainty = sigma
    bench = np.sqrt(8 * T0 / (np.pi * M))
    res = render_match(
        rfn, band, {"T": T0},
        Benchmark("mean molecular speed <v>", round(bench, 4), 0.08, "sqrt(8kT/pi m) Maxwell-Boltzmann (EXTERNAL)", "speed"),
        nulls=[("at absolute zero the molecules stop -- the mean speed vanishes (T->0 -> <v>->0)", {"T": 1e-4}, lambda v, m: v < m / 50)],
        perturbations=[("heating speeds the molecules up as sqrt(T) (T up -> faster)", {"T": 2.0}, lambda v, best: v > best)],
        notes=["the mean speed is sqrt(8kT/pi m); it vanishes at T=0 and grows as sqrt(T)"])
    print(res.report())
    # ★the three speeds ordering + the high-energy (Arrhenius) tail
    s = speeds(T0); vp = np.sqrt(2 * T0 / M); vmean = np.sqrt(8 * T0 / (np.pi * M)); vrms = np.sqrt(3 * T0 / M)
    vmean_mc, vrms_mc = s.mean(), np.sqrt((s ** 2).mean())
    Ea = 5.0; Ts = np.array([0.5, 0.7, 1.0, 1.5, 2.0])
    fr_mc = np.array([reactive_fraction(T, Ea) for T in Ts]); fr_an = gammaincc(1.5, Ea / Ts)
    slope = np.polyfit(1 / Ts, np.log(fr_mc), 1)[0]
    print(f"\n  T -> mean speed:  " + "  ".join(f"{t:.1f}:{mean_speed(t):.3f}" for t in (0.5, 1.0, 2.0)))
    print(f"  ★THREE SPEEDS (T=1): v_p={vp:.3f} < <v>={vmean_mc:.4f}(analytic {vmean:.4f}) < v_rms={vrms_mc:.4f}(analytic {vrms:.4f}) -- ordered 1:1.128:1.225")
    print(f"  ★ARRHENIUS TAIL: reactive fraction (KE>Ea={Ea}) vs upper-incomplete-gamma: " + "  ".join(f"T={t}:{f:.1e}/{a:.1e}" for t, f, a in zip(Ts, fr_mc, fr_an)))
    print(f"  Arrhenius slope d ln(fraction)/d(1/T) = {slope:.2f} (~ -Ea={-Ea}, plus the sqrt(Ea/kT) prefactor) -- the activation-energy law from COUNTING fast molecules")
    print(f"  (4) ★THE RATE LAW FROM A HEAD COUNT: sampling the speed distribution, the mean speed is sqrt(8kT/pi m) to {abs(vmean_mc-vmean)/vmean*100:.1f}% and the three speeds order v_p<<v><v_rms; the fraction with enough energy to react matches Gamma(3/2,Ea/kT)/Gamma(3/2) and falls as exp(-Ea/kT). The Arrhenius temperature dependence is not assumed -- it is the high-energy tail of the Maxwell distribution, the COLLISION-THEORY origin of the rate, distinct from kramers_escape's barrier-crossing MFPT route (two derivations, one law)")
    print(f"  (5) ★WHY A 10-DEGREE RISE ROUGHLY DOUBLES REACTION RATES: only the tail above Ea reacts, and exp(-Ea/kT) is exquisitely T-sensitive there -- near room temperature a 10 K rise multiplies the energetic fraction by ~2-3. The SAME tail sets which molecules escape a planet's gravity (Jeans escape -- hydrogen leaks, nitrogen stays), how fast isotopes separate, and why combustion has an ignition temperature; the distribution's WIDTH (the three speeds) is also what Doppler-broadens spectral lines in doppler_broadening")
    g4 = abs(vmean_mc - vmean) / vmean < 0.01 and (vp < vmean_mc < vrms_mc) and np.max(np.abs(fr_mc - fr_an) / fr_an) < 0.1   # mean; ordering; tail=incomplete-gamma
    g5 = mean_speed(1e-4) < bench / 50 and mean_speed(2.0) > mean_speed(0.5) and abs(mean_speed(4.0) / mean_speed(1.0) - 2.0) < 0.05  # null; rises; sqrt(T) scaling
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Maxwell-Boltzmann) — the molecular speed distribution and its Arrhenius tail:")
        print(f"  • the mean speed is {res.best:.4f}=sqrt(8kT/pi m) (band [{res.band_lo:.3f},{res.band_hi:.3f}]=temperature σ), to {abs(vmean_mc-vmean)/vmean*100:.1f}%.")
        print(f"  • the three speeds order v_p<<v><v_rms; the reactive tail matches the incomplete gamma and falls as exp(-Ea/kT).")
        print(f"  • ★the Arrhenius factor from counting energetic molecules -- the collision-theory rate law, the statistical sibling of Kramers.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, mean/order/tail {g4}, null/sqrt(T) {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
