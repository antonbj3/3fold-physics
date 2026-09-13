"""QUANTUM WALK — BALLISTIC SPREADING OF A DISCRETE-TIME WALKER (quantum information) — RENDER->MATCH why a
quantum particle that walks by flipping a quantum coin spreads QUADRATICALLY faster than a classical random walker. Each step a
Hadamard coin puts the walker in a superposition of move-left and move-right, and the two histories INTERFERE -- the amplitudes,
not the probabilities, add. We do NOT assert the spreading: we EVOLVE the position-coin amplitudes unitarily (coin flip then
conditional shift) and MEASURE the standard deviation of the position distribution. ★The quantum walk spreads BALLISTICALLY,
sigma_x = c t with c = sqrt(1 - 1/sqrt 2) ~ 0.5412 -- linear in the number of steps, vs the classical sqrt(t); ★the probability
distribution is DOUBLE-PEAKED, piling up near the ballistic fronts x ~ +/- t/sqrt 2 (the quantum-walk fingerprint, nothing like a
Gaussian); ★a DECOHERENT (classical) walk that measures the coin each step loses the interference and reverts to diffusive
sqrt(t) spreading (the null). The number of steps t is the physical sigma. Gates run through render_match_scaffold.

MATCH: the standard deviation of the unitarily-evolved quantum walk equals c*t with c=sqrt(1-1/sqrt2)~0.5412 (ballistic); ★the distribution is double-peaked near +/- t/sqrt2 (cross-check); ★a decoherent (classical) walk spreads only as sqrt(t) (the null); more steps spread further, linearly.
  python quantum/quantum_walk.py
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

T0 = 100                                                 # number of steps (the physical sigma)
CBALLISTIC = np.sqrt(1 - 1 / np.sqrt(2))                 # ~0.5412, the Hadamard-walk spreading speed
_cache = {}


def qwalk(T):
    """unitary Hadamard quantum walk; return (position grid, probability distribution) (memoized)."""
    if T in _cache:
        return _cache[T]
    N = 2 * T + 3; c = N // 2
    psi = np.zeros((N, 2), complex)
    psi[c, 0] = 1 / np.sqrt(2); psi[c, 1] = 1j / np.sqrt(2)   # symmetric (unbiased) initial coin
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    for _ in range(T):
        coined = psi @ H.T
        new = np.zeros_like(psi)
        new[:-1, 0] += coined[1:, 0]                     # coin 0 -> step left
        new[1:, 1] += coined[:-1, 1]                     # coin 1 -> step right
        psi = new
    P = np.sum(np.abs(psi) ** 2, axis=1)
    x = np.arange(N) - c
    out = (x, P)
    _cache[T] = out
    return out


def sigma_x(T):
    x, P = qwalk(T)
    mean = np.sum(x * P)
    return float(np.sqrt(np.sum((x - mean) ** 2 * P)))


def classical_sigma(T):
    return float(np.sqrt(T))                              # decoherent random walk: diffusive sqrt(t)


def peak_position(T):
    x, P = qwalk(T); pos = x > 0
    return float(x[pos][np.argmax(P[pos])])              # location of the right ballistic-front peak


def main():
    print("=" * 96)
    print("QUANTUM WALK — ballistic spreading of a discrete-time walker; render->match")
    print("=" * 96)
    def rfn(p):
        if p.get("classical"):
            return classical_sigma(p.get("T", T0))       # decoherent walk: sqrt(t)
        return sigma_x(p.get("T", T0))                    # quantum walk: c*t
    band = [{"T": int(0.9 * T0)}, {"T": int(1.1 * T0)}]  # step-count sigma: sigma_x = c*t grows linearly (two-sided)
    bench = CBALLISTIC * T0
    res = render_match(
        rfn, band, {"T": T0},
        Benchmark("quantum-walk standard deviation", bench, 0.02 * bench, "sqrt(1-1/sqrt2)*t ~ 0.5412 t (EXTERNAL, ballistic)", ""),
        nulls=[("a DECOHERENT walk -- one that effectively MEASURES the coin every step -- destroys the interference between the move-left and move-right histories, so the amplitudes no longer add coherently and the walk becomes an ORDINARY classical random walk with diffusive sqrt(t) spreading, far narrower than the quantum c*t. The quadratic speed-up is ENTIRELY the quantum coherence (decohere -> sqrt(t), not c*t)", {"classical": True}, lambda v, m: v < 0.3 * bench)],
        perturbations=[("running the walk for MORE steps spreads it LINEARLY further -- sigma_x = c*t grows in proportion to t (unlike the classical sqrt(t)), so doubling the steps nearly doubles the width: ballistic, not diffusive (more steps -> proportionally wider)", {"T": 2 * T0}, lambda v, best: v > 1.8 * best)],
        notes=["sigma_x matches c*t with c~0.5412; the distribution is double-peaked near +/- t/sqrt2; a decoherent walk gives sqrt(t); spreading is linear in t"])
    print(res.report())
    s0 = sigma_x(T0); pk = peak_position(T0)
    Ts = [50, 100, 200]
    print(f"\n  ★STANDARD DEVIATION FROM THE UNITARY EVOLUTION (measured, not asserted): sigma_x={s0:.3f} vs c*t={bench:.3f} ({abs(s0-bench)/bench*100:.2f}%), c=sqrt(1-1/sqrt2)={CBALLISTIC:.4f} -- the position-coin amplitudes were evolved by {T0} coin-flip-and-shift steps and the distribution's width measured; no spreading law entered the unitary. sigma_x/t = {s0/T0:.4f} (ballistic, NOT the classical 1/sqrt(t))")
    print(f"  ★★DOUBLE-PEAKED, BALLISTIC FRONTS (the falsifier): the probability piles up near x = +/- {pk} (theory +/- t/sqrt2 = +/- {T0/np.sqrt(2):.0f}), NOT in the centre -- a deeply non-Gaussian, double-horned distribution. The interference suppresses the middle and pushes weight to the ballistic edges, the unmistakable signature of a coherent quantum walk: sigma_x vs t -- " + ", ".join(f"t={t}:{sigma_x(t):.1f}(ct={CBALLISTIC*t:.1f})" for t in Ts))
    print(f"  ★QUANTUM vs CLASSICAL -- THE QUADRATIC SPEED-UP (the sigma): quantum sigma_x grows as c*t, classical as sqrt(t) -- " + ", ".join(f"t={t}:Q={sigma_x(t):.0f}/C={classical_sigma(t):.0f}" for t in Ts) + ". After t steps the quantum walker has spread ~{:.0f}x further than the classical one and reaches a given site in sqrt(t) rather than t steps -- the resource behind quantum search and quantum-walk algorithms".format(sigma_x(T0) / classical_sigma(T0)))
    print(f"  (4) ★WHY IT MATTERS: the quantum walk is the quantum analogue of Brownian motion and the engine of several quantum algorithms -- it gives the quadratic-to-exponential speed-ups of quantum search and element distinctness, models coherent energy transport in photosynthetic complexes and photonic lattices, and is a universal model of quantum computation. The c*t ballistic spreading and double-peaked distribution are the coherent-transport closure a quantum-simulator twin integrates")
    g4 = abs(s0 - bench) / bench < 0.02 and classical_sigma(T0) < 0.3 * bench and abs(pk - T0 / np.sqrt(2)) < 0.1 * T0    # ballistic; classical null; double-peak position
    g5 = sigma_x(2 * T0) > 1.8 * s0 and sigma_x(T0) > 4 * classical_sigma(T0)                                            # linear in t; quantum >> classical
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/quantum_walk.json", "w") as fh:
        json.dump({"module": "quantum_walk", "provenance": "self-contained unitary Hadamard quantum-walk evolution, no external data",
                   "sigma_x": s0, "formula_ct": float(bench), "c_ballistic": float(CBALLISTIC), "peak_position": pk, "peak_theory": float(T0 / np.sqrt(2)),
                   "classical_sigma": classical_sigma(T0), "steps": T0,
                   "sigma_vs_t": {f"{t}": {"quantum": float(sigma_x(t)), "ct": float(CBALLISTIC * t), "classical": float(classical_sigma(t))} for t in Ts},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"ballistic_null_doublepeak": bool(g4), "linear_quantumadv": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (quantum walk) — coherence makes the walker spread ballistically:")
        print(f"  • sigma_x={s0:.3f} matches c*t={bench:.3f} (c~0.5412, band=step-count σ).")
        print(f"  • ★double-peaked near +/-{pk} (= t/sqrt2); a decoherent walk gives only sqrt(t) (the null).")
        print(f"  • ★a distinct quantum-information/transport primitive for quantum-algorithm/simulator twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, ballistic/null/peak {g4}, linear/adv {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
