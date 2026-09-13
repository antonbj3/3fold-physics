"""TWO-STREAM INSTABILITY — TWO COUNTER-STREAMING BEAMS PUMP A PLASMA WAVE (plasma kinetics) — RENDER->MATCH the
growth of an electrostatic wave fed by two interpenetrating electron beams. When two cold electron populations stream through each other
at +/- v0, their free streaming energy resonates with a plasma oscillation and a Langmuir wave grows exponentially -- the seed of beam-
plasma heating, double layers, and electron-beam turbulence. We do NOT assert the growth rate: we solve the genuine cold dispersion
1 = omega_b^2/(omega - k v0)^2 + omega_b^2/(omega + k v0)^2 (each beam omega_b^2 = omega_p^2/2) as a QUARTIC in omega and read the growth
off its complex ROOTS. ★The genuine quartic-root growth peaks at exactly omega_p/sqrt(8) -- the famous maximum two-stream rate that
EMERGES from the roots, not imposed; ★the fastest-growing wave sits at k v0 = sqrt(3/8) omega_p, and the band is unstable only for
k v0 < omega_p (longer waves; the cross-checks); ★if the two beams CO-stream (same direction) the free energy vanishes and the roots go
real -- no instability (the null); ★a denser plasma grows faster (gamma ~ omega_p). The plasma frequency omega_p (~ sqrt(density)) is the
physical sigma. render_match_scaffold.

MATCH: the genuine quartic-root growth peaks at omega_p/sqrt(8); ★the peak sits at k v0 = sqrt(3/8) omega_p and the band kv0<omega_p is unstable (cross-checks); ★co-streaming beams are stable (the null).
  python em/two_stream_instability.py
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
import os
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from render_match_scaffold import Benchmark, render_match

WP = 1.0           # plasma frequency (~ sqrt(electron density))
V0 = 1.0           # beam streaming speed (+/- V0)


def growth_k(k, wp=WP, v0=V0, co=False):
    """genuine: max imaginary part of the cold dispersion's roots (the growth rate)."""
    wb2 = wp ** 2 / 2
    if co:                                                           # both beams +v0 -> single Doppler-shifted oscillation
        r = np.roots([1, -2 * k * v0, k ** 2 * v0 ** 2 - wp ** 2])
        return float(np.max(r.imag))
    a = (k * v0) ** 2
    r = np.roots([1, 0, -2 * (a + wb2), 0, a ** 2 - 2 * wb2 * a])    # quartic omega^4 - 2(a+wb^2)omega^2 + (a^2-2 wb^2 a)
    return float(np.max(r.imag))


def gamma_max(wp=WP, v0=V0, co=False):
    """genuine: maximum growth rate over wavenumber."""
    ks = np.linspace(0.01 * wp / v0, 1.3 * wp / v0, 700)
    return float(max(growth_k(k, wp, v0, co) for k in ks))


def growth_analytic(k, wp=WP, v0=V0):
    a = (k * v0) ** 2; wb2 = wp ** 2 / 2
    xm = (a + wb2) - np.sqrt(wb2) * np.sqrt(4 * a + wb2)
    return float(np.sqrt(max(-xm, 0)))


def main():
    print("=" * 96)
    print("TWO-STREAM INSTABILITY — two counter-streaming beams pump a plasma wave; render->match")
    print("=" * 96)
    def rfn(p):
        return gamma_max(wp=p.get("wp", WP), co=p.get("co", False))
    band = [{"wp": 0.92 * WP}, {"wp": 1.08 * WP}]                   # sigma = plasma frequency omega_p (~ sqrt density)
    bench = WP / np.sqrt(8)
    res = render_match(
        rfn, band, {"wp": WP},
        Benchmark("max growth rate", bench, 0.0, "omega_p/sqrt(8) (EXTERNAL: cold two-stream)", "omega_p"),
        nulls=[("if the two beams CO-stream (both moving the SAME way at +v0 instead of toward each other) the relative-streaming free energy that feeds the wave disappears: the dispersion collapses to a single Doppler-shifted plasma oscillation omega = k v0 +/- omega_p with REAL roots, and the growth rate is zero. The instability is powered by the COUNTER-streaming; align the beams and the plasma merely oscillates (co-streaming -> no growth)", {"co": True}, lambda v, m: v < 0.02 * bench)],
        perturbations=[("a DENSER plasma (omega_p x 1.5, i.e. 2.25x the electron density) grows the wave FASTER -- the maximum rate scales as gamma = omega_p/sqrt(8) ~ sqrt(density). The two-stream instability is therefore most violent in dense beam-plasma systems and fusion injectors, where it sets the timescale on which an injected beam dumps its energy into Langmuir turbulence (denser -> faster growth)", {"wp": 1.5 * WP}, lambda v, best: v > best)],
        notes=["the genuine quartic-root growth peaks at omega_p/sqrt(8); the peak sits at k v0 = sqrt(3/8) omega_p; the band kv0<omega_p is unstable; co-streaming is stable"])
    print(res.report())
    gm = gamma_max(); ks = np.linspace(0.01, 1.3, 700); g = np.array([growth_k(k) for k in ks]); kpk = ks[np.argmax(g)]
    kpk_an = np.sqrt(3 / 8) * WP / V0
    disp = {k: (growth_k(k), growth_analytic(k)) for k in (0.3, 0.612, 0.9, 1.0, 1.1)}
    g_co = gamma_max(co=True); g_dense = gamma_max(wp=1.5 * WP)
    print(f"\n  ★GROWTH = QUARTIC-ROOT OF THE DISPERSION (computed, not asserted): solving 1 = omega_b^2/(omega-kv0)^2 + omega_b^2/(omega+kv0)^2 as a quartic and reading the growth off its complex roots gives a maximum rate {gm:.4f} vs omega_p/sqrt(8) = {bench:.4f} ({abs(gm-bench)/bench*100:.1f}%). The famous 1/sqrt(8) coefficient emerges from the roots -- no growth formula imposed")
    print(f"  ★★PEAK AT k v0 = sqrt(3/8) omega_p, UNSTABLE FOR kv0 < omega_p (the falsifiers): the fastest-growing wave sits at k = {kpk:.4f} vs sqrt(3/8) omega_p/v0 = {kpk_an:.4f}, and the growth across k -- " + ", ".join(f"{k}:{s:.3f}/{a:.3f}" for k, (s, a) in disp.items()) + " (quartic/analytic) -- is nonzero only below kv0 = omega_p and vanishes above. Long waves resonate with the beams; short ones outrun them")
    print(f"  ★COUNTER-STREAMING IS THE ENGINE (cross-check / the null): align the beams (co-stream) and the max growth drops to {g_co:.2e} ~ 0 -- the roots become real, a plain oscillation. The free energy is the RELATIVE streaming; without it there is no instability. Double the density (omega_p x1.5) and the rate climbs to {g_dense:.3f} = 1.5 x faster")
    print(f"  (4) ★WHY IT MATTERS: the two-stream instability governs how an injected electron beam dumps energy into a plasma -- beam-plasma heating in fusion devices, the saturation of electron beams into Langmuir turbulence and double layers, type-III solar radio bursts, and the electron-beam diagnostics of space and lab plasmas. gamma_max = omega_p/sqrt(8) is the closure a beam-plasma twin validates against")
    g4 = res.ok and abs(gm - bench) / bench < 0.02 and abs(kpk - kpk_an) / kpk_an < 0.02 and all(abs(s - a) < 0.02 for s, a in disp.values())  # 1/sqrt8; k_peak; dispersion
    g5 = g_co < 0.02 * bench and g_dense > gm and disp[0.9][0] > 0 and disp[1.1][0] < 0.02 * bench                                              # co-stream null; denser faster; unstable range cutoff
    ok = g4 and g5
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/two_stream_instability.json", "w") as fh:
        json.dump({"module": "two_stream_instability", "provenance": "self-contained: quartic-root growth of the cold two-stream dispersion; max vs omega_p/sqrt(8)",
                   "wp": WP, "v0": V0, "gamma_max": gm, "gamma_max_analytic": bench, "k_peak": kpk, "k_peak_analytic": kpk_an,
                   "gamma_costream": g_co, "gamma_dense": g_dense, "dispersion": {str(k): {"quartic": s, "analytic": a} for k, (s, a) in disp.items()},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"coeff_kpeak_dispersion": bool(g4), "null_costream": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (two-stream instability) — two counter-streaming beams pump a plasma wave:")
        print(f"  • max growth = {gm:.4f} = omega_p/sqrt(8) (band=plasma-frequency σ); genuine quartic-root of the dispersion.")
        print(f"  • ★peak at k v0 = sqrt(3/8) omega_p; unstable for kv0<omega_p; co-streaming stable (null); denser faster.")
        print(f"  • ★a distinct plasma-kinetics primitive -- the two-stream closure; distinct from fluid Kelvin-Helmholtz + magnetic MRI.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, coeff/kpeak/disp {g4}, null/dense {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
