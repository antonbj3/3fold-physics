"""CURIE-WEISS MEAN-FIELD FERROMAGNET — SELF-CONSISTENT MAGNETIZATION + PHASE TRANSITION — RENDER->MATCH how a
lattice of spins spontaneously magnetizes below a critical temperature. In mean-field theory each spin feels the AVERAGE field of
its neighbors, so the magnetization obeys the SELF-CONSISTENT equation
  m = tanh( (J z m + h) / (k_B T) ),
with J the exchange, z the coordination, h the external field. We do NOT assert m: we SOLVE this fixed point. ★Above the Curie
temperature T_c = J z / k_B the only solution at h=0 is m=0 (paramagnet); BELOW T_c a nonzero spontaneous magnetization appears
(ferromagnet) -- a continuous (second-order) phase transition. ★The paramagnetic susceptibility follows the CURIE-WEISS law
chi = C/(T - T_c), diverging as T -> T_c+ -- the experimental fingerprint of a mean-field ferromagnet. The temperature T is the
physical sigma. render_match_scaffold.

MATCH: the paramagnetic susceptibility from the self-consistent mean-field equation equals the Curie-Weiss law C/(T-T_c) above T_c; ★below T_c a spontaneous magnetization appears (m>0 at h=0, the null/order-parameter falsifier the high-T phase lacks); the susceptibility diverges as T->T_c+; far above T_c it crosses over to the free-spin Curie law chi~1/T.
  python em/curie_weiss_ferromagnet.py
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
from scipy.optimize import brentq
from render_match_scaffold import Benchmark, render_match

TC = 1.0                                                 # Curie temperature J z / k_B (energy/temperature unit)
C = 1.0                                                  # Curie constant (single-spin)


def m_selfconsistent(T, h=0.0):
    """fixed point of m = tanh((TC*m + h)/T); the (stable) ferromagnetic branch for T<TC, the paramagnetic root otherwise."""
    f = lambda m: np.tanh((TC * m + h) / T) - m
    if h == 0.0 and T < TC:
        return float(brentq(f, 1e-9, 1.0))               # nontrivial spontaneous root
    lo, hi = (-1 + 1e-9, 1 - 1e-9)
    if f(lo) * f(hi) > 0:
        return 0.0
    return float(brentq(f, lo, hi))


def susceptibility(T, dh=1e-4):
    return (m_selfconsistent(T, dh) - m_selfconsistent(T, -dh)) / (2 * dh)   # chi = dm/dh at h=0


def main():
    print("=" * 96)
    print("CURIE-WEISS FERROMAGNET — self-consistent mean-field magnetization + transition; render->match")
    print("=" * 96)
    TTEST = 1.5 * TC                                     # paramagnetic test temperature
    band = [{"T": 1.4 * TC}, {"T": 1.6 * TC}]            # temperature sigma: chi grows as T->T_c (two-sided, above T_c)
    bench = C / (TTEST - TC)
    def rfn2(p):
        if p.get("spont"):
            return m_selfconsistent(0.6 * TC, 0.0)        # spontaneous magnetization below T_c
        return susceptibility(p.get("T", TTEST))
    res = render_match(
        rfn2, band, {"T": TTEST},
        Benchmark("paramagnetic susceptibility chi", bench, 0.03 * bench, "Curie-Weiss C/(T-T_c) (EXTERNAL, mean-field)", ""),
        nulls=[("BELOW the Curie temperature (T=0.6 T_c) the self-consistent equation has a NONZERO root at zero field -- a SPONTANEOUS magnetization (the order parameter) with no applied field, the defining feature of the ferromagnetic phase the paramagnet lacks (T<T_c -> spontaneous m>0)", {"spont": True}, lambda v, m: v > 0.5)],
        perturbations=[("APPROACHING T_c from above (T 1.5->1.1 T_c) the susceptibility DIVERGES -- chi=C/(T-T_c) blows up as T->T_c+ (closer to T_c -> larger chi)", {"T": 1.1 * TC}, lambda v, best: v > 2.0 * best)],
        notes=["paramagnetic chi matches Curie-Weiss C/(T-T_c); spontaneous m below T_c; chi diverges at T_c"])
    print(res.report())
    chi0 = susceptibility(TTEST)
    print(f"\n  ★SUSCEPTIBILITY FROM THE SELF-CONSISTENT FIELD (solved, not asserted): chi({TTEST/TC:.1f} T_c)={chi0:.4f} vs Curie-Weiss C/(T-T_c)={bench:.4f} ({abs(chi0-bench)/bench*100:.2f}%) -- the mean-field fixed point m=tanh((Jzm+h)/T) was solved and dm/dh measured; no susceptibility formula entered the solve")
    print(f"  ★★THE PHASE TRANSITION (the order-parameter falsifier): spontaneous magnetization m(h=0) -- " + ", ".join(f"T/Tc={t}:{m_selfconsistent(t*TC):.3f}" for t in (0.5, 0.8, 0.95, 1.0, 1.2)) + ". A nonzero m appears for T<T_c (ferromagnet) and is EXACTLY zero for T>=T_c (paramagnet): a continuous (second-order) transition with the mean-field exponent m~(T_c-T)^{1/2} near T_c")
    print(f"  ★CURIE-WEISS DIVERGENCE (the sigma): inverse susceptibility 1/chi is LINEAR in T above T_c -- " + ", ".join(f"T/Tc={t}:1/chi={1/susceptibility(t*TC):.2f}" for t in (1.1, 1.5, 2.0, 3.0)) + " (= T-T_c), extrapolating to zero AT T_c. This straight 1/chi vs T line, and its x-intercept T_c, is how the Curie temperature is read from experiment")
    print(f"  (4) ★WHY IT MATTERS: the Curie-Weiss mean-field model is the prototype of every continuous phase transition -- it gives the spontaneous magnetization (order parameter), the diverging susceptibility and the critical temperature of a ferromagnet, and the same self-consistent structure underlies superconductivity (BCS), liquid crystals and alloy ordering. The order-parameter + susceptibility closure a magnetic/spintronic twin integrates")
    msp = m_selfconsistent(0.6 * TC); mhi = m_selfconsistent(1.2 * TC)
    g4 = abs(chi0 - bench) / bench < 0.03 and msp > 0.5 and abs(mhi) < 1e-6                         # chi; spontaneous below; zero above
    g5 = susceptibility(1.1 * TC) > 2.0 * chi0 and abs(1 / susceptibility(2.0 * TC) - (2.0 - 1.0) * TC) < 0.05   # divergence; linear 1/chi
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/curie_weiss_ferromagnet.json", "w") as fh:
        json.dump({"module": "curie_weiss_ferromagnet", "provenance": "self-contained self-consistent mean-field magnetization, no external data",
                   "chi_param": float(chi0), "curie_weiss_formula": float(bench), "Tc": TC, "m_spontaneous_06Tc": float(msp), "m_above_Tc": float(mhi),
                   "m_vs_T": {f"{t:.2f}": float(m_selfconsistent(t * TC)) for t in (0.5, 0.8, 0.95, 1.2)},
                   "inv_chi_vs_T": {f"{t:.2f}": float(1 / susceptibility(t * TC)) for t in (1.1, 1.5, 2.0)},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"chi_spont_zero": bool(g4), "divergence_linear": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Curie-Weiss ferromagnet) — spontaneous order and a diverging susceptibility:")
        print(f"  • chi({TTEST/TC:.1f} T_c)={chi0:.4f} matches Curie-Weiss C/(T-T_c)={bench:.4f} (band=temperature σ).")
        print(f"  • ★spontaneous m={msp:.3f} below T_c (zero above); chi diverges at T_c; 1/chi linear in T.")
        print(f"  • ★a distinct phase-transition/statistical-mechanics primitive for magnetic/spintronic twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, chi/spont/zero {g4}, divergence/linear {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
