"""LASER SPECKLE — INTENSITY STATISTICS OF COHERENT RANDOM INTERFERENCE (statistical optics) — RENDER->MATCH the
grainy intensity pattern coherent light forms after scattering off a rough surface, and how AVERAGING independent patterns smooths
it. Each detector point sums many scattered wavelets with RANDOM phases; the resultant is a 2D random walk in the complex plane, so
the intensity I=|sum e^{i phi}|^2 is EXPONENTIALLY distributed and its contrast C = std(I)/mean(I) = 1 (fully developed speckle).
Adding M INDEPENDENT speckle patterns (different polarizations, wavelengths, or angles) averages by the central-limit theorem and
the contrast DROPS as 1/sqrt(M). We do NOT assert this: we SIMULATE random-phasor sums and MEASURE the contrast and the intensity
moments. ★The M-averaged contrast matches 1/sqrt(M); ★a single pattern (M=1) is fully developed -- exponential intensity with
<I^2>/<I>^2 = 2 (the cross-check); ★removing the phase randomness (coherent, aligned phasors) gives a uniform field with NO speckle
(the null). The number of averaged patterns M is the physical sigma. render_match_scaffold. NIGHT (a distinct statistical-optics
primitive -- the coherent-noise closure a lidar / OCT / holography / laser-projection digital twin needs). Uses render_match_scaffold.

MATCH: the M-averaged speckle contrast from random-phasor sums equals 1/sqrt(M); ★a single fully-developed pattern has exponential statistics, <I^2>/<I>^2 = 2 (cross-check); ★aligning the phases (no randomness) removes the speckle entirely (the null); averaging more patterns lowers the contrast.
Reference: Goodman, Speckle Phenomena in Optics (fully developed speckle: exponential intensity, contrast 1, 1/sqrt(M) averaging law).
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
from render_match_scaffold import Benchmark, render_match

M0 = 4                                                   # number of averaged speckle patterns (the physical sigma)
NPX = 80_000
NPH = 120                                                # scatterers per detector point
RNG = np.random.default_rng(0)
_cache = {}


def contrast(M, coherent=False):
    """average M independent fully-developed speckle patterns; return the contrast std(I)/mean(I) (memoized)."""
    key = (M, coherent)
    if key in _cache:
        return _cache[key]
    I = np.zeros(NPX)
    for _ in range(M):
        ph = np.zeros((NPX, NPH)) if coherent else RNG.uniform(0, 2 * np.pi, (NPX, NPH))
        I += np.abs(np.sum(np.exp(1j * ph), axis=1)) ** 2
    I /= M
    c = float(np.std(I) / np.mean(I))
    _cache[key] = c
    return c


_sm_cache = {}


def second_moment(M=1):
    if M in _sm_cache:
        return _sm_cache[M]
    I = np.abs(np.sum(np.exp(1j * RNG.uniform(0, 2 * np.pi, (NPX, NPH))), axis=1)) ** 2
    v = float(np.mean(I ** 2) / np.mean(I) ** 2)         # <I^2>/<I>^2 (=2 for exponential)
    _sm_cache[M] = v
    return v


def main():
    print("=" * 96)
    print("LASER SPECKLE — contrast and intensity statistics of random interference; render->match")
    print("=" * 96)
    def rfn(p):
        if p.get("coherent"):
            return contrast(1, coherent=True)            # aligned phases -> no speckle
        return contrast(p.get("M", M0))                  # M-averaged contrast
    band = [{"M": 5}, {"M": 3}]                          # averaging sigma: contrast 1/sqrt(M) (integer band brackets M=4)
    bench = 1 / np.sqrt(M0)
    res = render_match(
        rfn, band, {"M": M0},
        Benchmark("M-averaged speckle contrast", bench, 0.04 * bench, "1/sqrt(M) (EXTERNAL, speckle averaging)", ""),
        nulls=[("if the scattered wavelets arrive with ALIGNED phases (no surface roughness, a perfectly coherent in-phase sum) the resultant is deterministic -- every detector point gets the same intensity, std(I)=0 and there is NO speckle. The graininess is ENTIRELY the random-phase interference; remove the randomness and it vanishes (aligned phases -> contrast 0)", {"coherent": True}, lambda v, m: v < 0.02)],
        perturbations=[("averaging MORE independent speckle patterns (larger M) lets the central-limit theorem smooth the intensity, so the contrast DROPS as 1/sqrt(M) -- more diversity (polarization, wavelength, angle) washes out the grain (more patterns -> lower contrast)", {"M": 16}, lambda v, best: v < 0.6 * best)],
        notes=["the M-averaged contrast matches 1/sqrt(M); a single pattern is exponential with <I^2>/<I>^2=2; aligned phases give no speckle; more averaging lowers contrast"])
    print(res.report())
    c0 = contrast(M0); sm = second_moment(1); c1 = contrast(1)
    Ms = [1, 4, 9, 16]
    print(f"\n  ★CONTRAST FROM RANDOM-PHASOR SUMS (measured, not asserted): C(M={M0})={c0:.4f} vs 1/sqrt(M)={bench:.4f} ({abs(c0-bench)/bench*100:.1f}%) -- {NPX//1000}k detector points each summing {NPH} random-phase wavelets were built and the contrast measured; no statistics formula entered the simulation")
    print(f"  ★★FULLY-DEVELOPED SPECKLE IS EXPONENTIAL (cross-check): a single pattern (M=1) has contrast {c1:.3f} (=1) and normalized second moment <I^2>/<I>^2={sm:.3f} (=2, the exponential signature) -- the intensity is a random walk in the complex plane whose squared length is exponentially distributed: lots of dark points, a long bright tail. This is why coherent images are grainy and why the brightest speckles are ~e times the mean")
    print(f"  ★CONTRAST ~ 1/sqrt(M) (the sigma): contrast vs averaged patterns -- " + ", ".join(f"M={m}:{contrast(m):.3f}(1/sqrt={1/np.sqrt(m):.3f})" for m in Ms) + " -- each independent diversity channel (polarization, wavelength, angular subaperture) adds an incoherent average; M of them cut the contrast by sqrt(M), the standard speckle-reduction law")
    print(f"  (4) ★WHY IT MATTERS: speckle is the fundamental coherent noise of every laser-illuminated system -- it limits lidar and synthetic-aperture-radar resolution, the axial precision of OCT, the fidelity of holography and laser projection, and it is the very signal in speckle metrology and laser-Doppler flowmetry. The 1/sqrt(M) reduction and exponential statistics are the coherent-noise closure a structured-illumination twin integrates")
    g4 = abs(c0 - bench) / bench < 0.04 and abs(c1 - 1.0) < 0.03 and abs(sm - 2.0) < 0.08                       # 1/sqrt(M); contrast 1; exponential 2nd moment
    g5 = contrast(16) < 0.6 * c0 and contrast(1, coherent=True) < 0.02                                          # more averaging lower; coherent null
    ok = res.ok and g4 and g5
    import os, json
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "laser_speckle.json"), "w") as fh:
        json.dump({"module": "laser_speckle", "provenance": "self-contained random-phasor speckle simulation, no external data",
                   "contrast_M4": c0, "formula_inv_sqrt_M": float(bench), "contrast_M1": c1, "second_moment_M1": sm, "M": M0,
                   "contrast_coherent": float(contrast(1, coherent=True)),
                   "contrast_vs_M": {f"{m}": {"measured": float(contrast(m)), "theory": float(1 / np.sqrt(m))} for m in Ms},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"contrast_exp_moment": bool(g4), "averaging_coherent": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (laser speckle) — coherent random interference is exponential, averaging smooths it:")
        print(f"  • contrast C(M={M0})={c0:.4f} matches 1/sqrt(M)={bench:.4f} (band=averaging σ).")
        print(f"  • ★a single pattern is exponential (<I^2>/<I>^2={sm:.2f}=2); aligned phases give no speckle (the null).")
        print(f"  • ★a distinct statistical-optics primitive for lidar/OCT/holography/laser-projection twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, contrast/exp {g4}, averaging/coherent {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
