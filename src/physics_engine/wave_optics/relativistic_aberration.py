"""Relativistic aberration and beaming (the headlight effect): render->match of why a source that radiates
isotropically in its own rest frame appears, to a lab observer it moves toward, to beam its light into a narrow
forward cone.

Photon directions transform between frames by the aberration law
    cos theta_lab = (cos theta' + beta) / (1 + beta cos theta'),
so the rest-frame median ray (theta' = 90 deg) tips forward to theta_lab = arccos(beta). The beaming is not asserted:
an isotropic rest-frame distribution is sampled (cos theta' uniform on [-1, 1]), every photon is transformed, and the
lab-frame angular distribution is measured. The median emission angle is arccos(beta) and the forward-radiated
fraction is (1 + beta)/2; ultra-relativistically half the light falls inside a cone of half-angle ~1/gamma; at
beta = 0 the distribution is isotropic again (the null). The source speed beta is the physical uncertainty band (c = 1).

INPUT: none (Monte-Carlo, fixed seed). OUTPUT: printed gate lines + `artifacts/relativistic_aberration.json`.
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

BETA0 = 0.5                                              # source speed (the physical sigma); c=1
NPHOT = 2_000_000
RNG = np.random.default_rng(0)


def lab_costheta(beta):
    ct = RNG.uniform(-1, 1, NPHOT)                       # isotropic rest frame: cos theta' uniform
    return (ct + beta) / (1 + beta * ct)                 # relativistic aberration


def median_angle(beta):
    return float(np.degrees(np.arccos(np.median(lab_costheta(beta)))))


def forward_fraction(beta):
    return float(np.mean(lab_costheta(beta) > 0))        # fraction radiated into the forward hemisphere


def main():
    print("=" * 96)
    print("RELATIVISTIC ABERRATION / BEAMING — the headlight effect; render->match")
    print("=" * 96)
    def rfn(p):
        return median_angle(p.get("b", BETA0))           # median lab-frame emission angle
    band = [{"b": 1.1 * BETA0}, {"b": 0.9 * BETA0}]      # source-speed sigma: median angle decreases with beta (two-sided)
    bench = float(np.degrees(np.arccos(BETA0)))
    res = render_match(
        rfn, band, {"b": BETA0},
        Benchmark("median lab-frame emission angle", bench, 0.03 * bench, "arccos(beta) (EXTERNAL, aberration)", "deg"),
        nulls=[("a STATIONARY source (beta=0) radiates the SAME in the lab as in its rest frame -- isotropic, with the median ray at 90 degrees and exactly half the light in each hemisphere: there is no preferred direction and no beaming. The headlight effect is entirely the aberration of a MOVING source (beta=0 -> median 90 deg)", {"b": 1e-4}, lambda v, m: abs(v - 90) < 1.0)],
        perturbations=[("a FASTER source aberrates its light more strongly, so the median ray tips further FORWARD -- the median angle arccos(beta) shrinks toward 0 and the cone narrows (more speed -> sharper forward beam, the relativistic headlight)", {"b": 0.9}, lambda v, best: v < 0.6 * best)],
        notes=["the median aberrated angle matches arccos(beta); the forward fraction is (1+beta)/2; beta=0 is isotropic; faster beams more sharply"])
    print(res.report())
    med0 = median_angle(BETA0); ff0 = forward_fraction(BETA0)
    bs = [0.0, 0.3, 0.6, 0.9]
    g = 1 / np.sqrt(1 - BETA0 ** 2)
    print(f"\n  ★MEDIAN ANGLE FROM THE TRANSFORMED PHOTONS (measured, not asserted): median={med0:.2f} deg vs arccos(beta)={bench:.2f} ({abs(med0-bench)/bench*100:.2f}%) -- {NPHOT//1000000}M isotropic rest-frame photons were aberrated through cos theta_lab=(cos theta'+beta)/(1+beta cos theta') and the median taken; no beaming formula entered the transform")
    print(f"  ★★FORWARD FRACTION = (1+beta)/2 (the cross-check): {ff0:.3f} vs (1+beta)/2={ (1+BETA0)/2:.3f} -- a moving source radiates MORE than half its light forward; the fraction rises linearly from 1/2 (isotropic) toward 1 (all forward) as beta->1. forward fraction vs beta -- " + ", ".join(f"b={b}:{forward_fraction(b):.2f}" for b in bs) + " (the headlight filling the forward cone)")
    print(f"  ★MEDIAN ANGLE -> arccos(beta), CONE ~1/gamma (the sigma): median vs beta -- " + ", ".join(f"b={b}:{median_angle(b):.1f}deg(arccos {np.degrees(np.arccos(b)):.1f})" for b in bs) + f". At beta={BETA0} the relativistic half-angle 1/gamma={np.degrees(1/g):.1f} deg; ultra-relativistically half the light falls inside ~1/gamma -- the beam a synchrotron/jet appears to sweep")
    print(f"  (4) ★WHY IT MATTERS: relativistic beaming sets the APPARENT brightness and visibility of everything that moves near c -- the one-sided jets of blazars and microquasars (Doppler boosting), the forward-peaked lobes of synchrotron radiation in storage rings and free-electron lasers, and the angular flux a relativistic-source twin must integrate. The arccos(beta) median and (1+beta)/2 forward fraction are the angular-beaming closure")
    g4 = abs(med0 - bench) / bench < 0.03 and abs(ff0 - (1 + BETA0) / 2) < 0.01 and abs(median_angle(1e-4) - 90) < 1.0   # median; forward fraction; isotropic null
    g5 = median_angle(0.9) < 0.6 * med0 and forward_fraction(0.9) > forward_fraction(0.3)                                # faster beams more; forward fraction grows
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/relativistic_aberration.json", "w") as fh:
        json.dump({"module": "relativistic_aberration", "provenance": "self-contained Monte-Carlo aberration of an isotropic source, no external data",
                   "median_angle": med0, "formula_arccos_beta": bench, "forward_fraction": ff0, "forward_fraction_formula": float((1 + BETA0) / 2),
                   "half_angle_inv_gamma_deg": float(np.degrees(1 / g)), "beta": BETA0,
                   "vs_beta": {f"{b}": {"median": float(median_angle(b)), "arccos": float(np.degrees(np.arccos(b))), "forward_frac": float(forward_fraction(b))} for b in bs},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"median_fraction_null": bool(g4), "faster_fracgrows": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (relativistic aberration) — a moving isotropic source beams forward:")
        print(f"  • median angle={med0:.2f} deg matches arccos(beta)={bench:.2f} (band=source-speed σ).")
        print(f"  • ★forward fraction {ff0:.3f}=(1+beta)/2; beta=0 isotropic (the null); cone ~1/gamma ultra-relativistically.")
        print(f"  • ★a distinct relativistic-kinematics primitive for jet/synchrotron/blazar twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, median/fraction/null {g4}, faster/fracgrows {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
