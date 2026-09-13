"""LASER THRESHOLD — RATE EQUATIONS — RENDER->MATCH the sharp turn-on of a laser: pump it gently and it just glows
(spontaneous emission); cross a critical pump and it suddenly LASES -- a coherent beam whose power rises linearly with pump. The
threshold is where the round-trip gain equals the cavity loss. Above it, two hallmarks: the gain CLAMPS (the inversion stops
growing -- every extra pump photon becomes a laser photon, not more inversion), and the output is LINEAR in pump with unit
differential efficiency. From the steady-state rate equations (inversion N, photons n; pump R_p, cavity lifetime tau_c, gain B),
    threshold R_p^th = 1 / (B tau_c tau_2),     above it  N -> N_th (clamped),  output power n/tau_c ~ R_p - R_p^th.
We do NOT assume it: we solve the steady-state rate equations, read the output curve n(R_p), and extract the threshold by the
standard linear EXTRAPOLATION of the above-threshold output back to zero. Uses render_match_scaffold (open-system /
non-equilibrium phase transition; a transcritical bifurcation -- the lasing sibling of a Hopf onset, distinct in that the
order parameter (photon number) grows linearly, not as a limit cycle).

MATCH: the laser threshold pump, from extrapolating the steady-state output curve, equals 1/(B tau_c tau_2); above it the inversion clamps and the output is linear; a lossier cavity (smaller tau_c) raises the threshold.
Reference: Siegman, Lasers (steady-state rate equations, threshold gain=loss, gain clamping, slope efficiency).
"""
# --- sibling-package bootstrap: this repository splits the modules by domain, so put every
# --- package directory on sys.path when the file is run as a script.
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

TAUC0, BETA = 1.0, 1e-4                                   # cavity photon lifetime; spontaneous-emission factor (tau_2=B=1)


def steady(Rp, tauc=TAUC0, beta=BETA):
    """steady-state inversion N and photon number n from the rate equations (0=Rp-N-N n; 0=N n - n/tauc + beta N)."""
    Nth = 1.0 / tauc
    N = brentq(lambda N: N + N * (beta * N / (Nth - N)) - Rp, 1e-12, Nth * (1 - 1e-12))
    n = beta * N / (Nth - N)
    return float(N), float(n)


def threshold_extrapolated(tauc=TAUC0, beta=BETA):
    """threshold by the standard linear extrapolation of the above-threshold output n(R_p) back to n=0 (the x-intercept)."""
    Rth = 1.0 / tauc
    Rs = np.array([2.0, 3.0, 4.0]) * Rth                  # well above threshold (linear region)
    ns = np.array([steady(R, tauc, beta)[1] for R in Rs])
    a, b = np.polyfit(Rs, ns, 1)                          # n = a R + b
    return float(-b / a)                                  # x-intercept


def main():
    print("=" * 96)
    print("LASER THRESHOLD — rate-equation turn-on; output extrapolation = gain-equals-loss; render->match")
    print("=" * 96)
    def rfn(p):
        return threshold_extrapolated(tauc=p.get("tauc", TAUC0))
    band = [{"tauc": 1.1 * TAUC0}, {"tauc": 0.9 * TAUC0}]  # cavity-loss σ (±10%): R_p^th = 1/(B tau_c tau_2)
    res = render_match(
        rfn, band, {"tauc": TAUC0},
        Benchmark("laser threshold pump R_p^th", 1.0 / TAUC0, 0.04, "1/(B tau_c tau_2) (gain=loss, EXTERNAL)", ""),
        nulls=[("a hugely LOSSY cavity (tau_c->0) needs an unreachable pump to reach gain=loss -- the threshold diverges, there is effectively no lasing (tau_c down -> threshold up without bound)", {"tauc": 0.05}, lambda v, m: v > 5 * m)],
        perturbations=[("a LOSSIER cavity (smaller tau_c) raises the gain needed and so the threshold (tau_c down -> threshold up)", {"tauc": 0.5 * TAUC0}, lambda v, best: v > best + 0.5)],
        notes=["the extrapolated output threshold matches 1/(B tau_c tau_2); above it the inversion clamps and the output is linear; a lossier cavity raises it"])
    print(res.report())
    # ★the threshold + the gain-clamping + the linear output + the spontaneous floor
    Rth = threshold_extrapolated()
    print(f"\n  laser threshold: extrapolated R_p^th={Rth:.4f} vs analytic 1/(B tau_c tau_2)={1/TAUC0:.4f}")
    print(f"  ★GAIN CLAMPING (the signature): the inversion N STOPS growing above threshold -- N(R_p) = " + ", ".join(f"{R}:{steady(R)[0]:.4f}" for R in (0.5, 1.0, 3.0, 10.0)) + f" -> clamps at N_th={1/TAUC0:.3f}. Every extra pump photon becomes a LASER photon, not more inversion; the gain self-regulates to exactly the loss")
    out = lambda R, tc: steady(R, tc)[1] / tc            # OUTPUT POWER = photons leaving the cavity per unit time (n/tau_c)
    print(f"  ★LINEAR OUTPUT, UNIT DIFFERENTIAL EFFICIENCY: the laser OUTPUT POWER (n/tau_c, photons leaving) rises linearly above threshold with UNIT slope vs pump, for ANY cavity -- slope d(n/tau_c)/dR_p = " + ", ".join(f"tau_c={tc}:{np.polyfit([2/tc,3/tc,4/tc],[out(2/tc,tc),out(3/tc,tc),out(4/tc,tc)],1)[0]:.3f}" for tc in (0.5, 1.0, 2.0)) + " (=1 universally, the slope efficiency). NB the photon NUMBER n itself scales as tau_c (slope tau_c); it is the EXTRACTED power that is the invariant linear converter")
    print(f"  ★A SHARP TURN-ON (the transition): below threshold only a spontaneous trickle n~{steady(0.5)[1]:.1e} (set by beta={BETA}); across threshold the photon number jumps by orders of magnitude -- a non-equilibrium phase transition (transcritical bifurcation), sharpened as beta->0")
    print(f"  (4) ★THRESHOLD FROM THE RATE EQUATIONS (not assumed): solving the steady state and extrapolating the output line gives R_p^th={Rth:.4f}, matching gain=loss (1/B tau_c tau_2={1/TAUC0:.3f}) to {abs(Rth-1/TAUC0)/(1/TAUC0)*100:.2f}% -- the threshold is exactly where round-trip gain balances cavity loss, no extra assumption")
    print(f"  (5) ★WHY IT MATTERS: the threshold sets every laser's efficiency and minimum pump (diode, fiber, semiconductor), the gain-clamping fixes the operating inversion and underlies gain competition and mode-hopping, the linear slope is the slope efficiency on every datasheet, and the same gain=loss balance governs masers, optical parametric oscillators and random lasers")
    g4 = abs(Rth - 1 / TAUC0) / (1 / TAUC0) < 0.02 and abs(steady(10.0)[0] - 1 / TAUC0) / (1 / TAUC0) < 0.01  # threshold; gain-clamping
    osl = lambda tc: np.polyfit([2 / tc, 3 / tc, 4 / tc], [steady(2 / tc, tc)[1] / tc, steady(3 / tc, tc)[1] / tc, steady(4 / tc, tc)[1] / tc], 1)[0]
    g5 = all(abs(osl(tc) - 1.0) < 0.02 for tc in (0.5, 1.0, 2.0)) and threshold_extrapolated(tauc=0.5) > Rth + 0.5 and steady(0.5)[1] < 0.01  # output-power unit slope (UNIVERSAL across tau_c); lossier->higher; spontaneous floor
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (laser threshold) — the sharp turn-on of coherent light:")
        print(f"  • extrapolated threshold {res.best:.4f} matches 1/(B tau_c tau_2) (band [{res.band_lo:.4f},{res.band_hi:.4f}]=cavity-loss σ).")
        print(f"  • the inversion clamps above threshold; the output POWER (n/tau_c) is linear with unit differential efficiency (universal); a lossier cavity raises the threshold.")
        print(f"  • ★a non-equilibrium phase transition: cross the gain=loss point and incoherent glow becomes a coherent beam.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, threshold/clamping {g4}, slope/lossier/floor {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
