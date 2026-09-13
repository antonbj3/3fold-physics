"""PLASMA FREQUENCY — RENDER→MATCH why you can hear a shortwave station from the other side of the planet but a satellite
needs GHz, and how an ionosonde weighs the ionosphere from the ground. Free electrons in a plasma oscillate at a natural
frequency set only by their density; an EM wave below it is REFLECTED (the electrons keep up and screen the field), above it
PASSES THROUGH:
    f_p = (1/2π)·√(n·e²/(ε₀·m_e)) ≈ 8.98·√n  Hz   (n in m⁻³)
So the F-layer (n≈10¹² m⁻³) reflects HF up to ~9 MHz — the skip that carries shortwave over the horizon — but is transparent to
VHF/GHz (satellites, GPS). Uses render_match_scaffold.

MATCH: the F2-layer critical frequency (foF2) is ~9 MHz at a daytime density of 10¹² m⁻³; f_p∝√n, so the ionosonde reads
density off reflected frequency. render→match, never fit: e, ε₀, m_e are fundamental; only the electron density is the σ.

  python em/plasma_frequency.py
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

E, EPS0, ME = 1.602176634e-19, 8.8541878128e-12, 9.1093837015e-31


def fp_MHz(n):
    return 1.0 / (2 * np.pi) * np.sqrt(n * E ** 2 / (EPS0 * ME)) / 1e6


def main():
    print("=" * 92)
    print("PLASMA FREQUENCY — f_p≈8.98√n Hz; ionospheric radio reflection; render→match")
    print("=" * 92)
    n_F = 1e12
    def rfn(p):
        return fp_MHz(p["n"])
    band = [{"n": 0.80e12}, {"n": 1.25e12}]                        # F-layer daytime electron density = σ
    res = render_match(
        rfn, band, {"n": n_F},
        Benchmark("F2-layer critical frequency foF2", 9.0, 1.0, "ionosonde daytime ~5-15 MHz", "MHz"),
        nulls=[("vacuum (n→0, no plasma)", {"n": 1.0}, lambda f, m: f < m / 100)],
        perturbations=[("denser plasma (n↑)", {"n": 4e12}, lambda f, best: f > best)],
        notes=["no electrons → no plasma oscillation → no reflection; a denser layer reflects higher frequencies"])
    print(res.report())
    # ★the reflect/pass split + the √n ionosonde + the layer ladder
    layers = [("D (day)", 1e9), ("E", 1e11), ("F1", 3e11), ("F2 (day)", 1e12)]
    print("\n  ionospheric layer → electron density → plasma frequency:")
    for nm, n in layers:
        print(f"    {nm:9s} n={n:.0e} m⁻³ → f_p={fp_MHz(n):.2f} MHz")
    ns = np.array([1e11, 1e12]); slope = np.polyfit(np.log(ns), np.log([fp_MHz(n) for n in ns]), 1)[0]
    f_gps = 1575.42                                                 # GPS L1 [MHz]
    print(f"  (4) ★REFLECT / PASS: HF below f_p={fp_MHz(n_F):.1f} MHz bounces off the F-layer (shortwave skip, over-the-horizon); VHF/GHz (GPS {f_gps:.0f} MHz ≫ f_p) passes straight through to satellites")
    print(f"  (5) ★IONOSONDE: f_p ∝ n^{slope:.2f} (√n) — sweep frequency, read the reflection cutoff, invert for density: the ground-based ionosphere probe")
    g4 = fp_MHz(n_F) < f_gps and fp_MHz(1e9) < 0.5                  # F-layer reflects HF, passes GPS; D-layer low
    g5 = abs(slope - 0.5) < 0.01
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (plasma frequency) — a density-set cutoff, predicted not fitted:")
        print(f"  • f_p=8.98√n renders {res.best:.1f} MHz for the F-layer (band [{res.band_lo:.1f},{res.band_hi:.1f}]=density σ); ionosondes read foF2≈9.")
        print(f"  • below it HF reflects (the worldwide shortwave skip), above it passes (GPS at {f_gps:.0f} MHz ≫ f_p reaches the satellite) — one cutoff, two regimes.")
        print(f"  • f_p∝√n turns a frequency sweep into a density profile; the same plasma cutoff sets radio blackout on re-entry and the solar-wind screen.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, reflect/pass {g4}, √n-slope {slope:.2f}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
