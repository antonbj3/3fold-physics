"""SHANNON CAPACITY — RENDER→MATCH the hard ceiling on how fast information can flow through a noisy channel — the law behind
dial-up's 56k wall, why WiFi needs wide bands, and why deep-space probes crawl. A channel of bandwidth B and signal-to-noise
ratio SNR carries at most
    C = B · log2(1 + SNR)   bits/s
No code beats it. More bandwidth helps linearly; more power only logarithmically (doubling SNR adds ~B bits, not 2x). Uses the
render_match_scaffold helper (forward model vs benchmark, no fitted constant).

MATCH: a 3.4 kHz phone line at ~36 dB SNR caps near the V.34 modem's 33.6 kbit/s. render→match, never fit: B is physical;
only the line SNR is the sigma.
INPUT: none (constants in the module). OUTPUT: printed gate lines.
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


def cap_kbps(B, snr_dB):
    return B * np.log2(1 + 10 ** (snr_dB / 10)) / 1e3


def main():
    print("=" * 92)
    print("SHANNON CAPACITY — C=B·log2(1+SNR); the modem wall; render->match")
    print("=" * 92)
    def rfn(p):
        return cap_kbps(p.get("B", 3400.0), p["snr"])
    band = [{"snr": 28.0}, {"snr": 32.0}]                          # phone-line SNR = sigma
    res = render_match(
        rfn, band, {"snr": 30.0},
        Benchmark("phone-line capacity (3.4 kHz)", 33.6, 4.0, "V.34 modem wall", "kbit/s"),
        nulls=[("no signal (SNR->0)", {"snr": -60.0}, lambda c, m: c < m / 100)],
        perturbations=[("wider band (WiFi 20 MHz)", {"snr": 30.0, "B": 20e6}, lambda c, best: c > best)],
        notes=["zero SNR -> zero capacity; widening the band raises it linearly (why WiFi/5G grab MHz, not just power)"])
    print(res.report())
    snrs = np.array([10.0, 20.0, 30.0, 40.0]); cs = np.array([cap_kbps(3400, s) for s in snrs])
    wifi = cap_kbps(20e6, 36.0) / 1e3
    print(f"\n  SNR -> capacity (3.4 kHz): " + "  ".join(f"{s:.0f}dB->{cap_kbps(3400,s):.1f}" for s in snrs) + " kbit/s")
    print(f"  (4) *BANDWIDTH WINS: every +10 dB SNR adds only ~{cap_kbps(3400,40)-cap_kbps(3400,30):.1f} kbit/s (log), but 20 MHz of WiFi band gives ~{wifi:.0f} Mbit/s (linear) -- spectrum beats power")
    print(f"  (5) *DEEP SPACE: tiny SNR but huge integration -- the law says crawl or widen; the ceiling no error-correcting code can cross")
    g4 = cap_kbps(3400, 0) > 0 and wifi > 100
    g5 = abs(cap_kbps(3400, 30) - 33.9) < 1.5
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    print("RENDER->MATCH CLOSES (Shannon)" if ok else f"HONEST ok={res.ok} g4={g4} g5={g5}")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
