"""ANTENNA APERTURE GAIN — RENDER->MATCH the gain of a dish from its size and frequency. An aperture antenna concentrates
power into a beam; its gain is the physical area in wavelengths, times an efficiency:
    G = eta (pi D / lambda)^2
So gain climbs with the SQUARE of diameter and of frequency, and the beam narrows as ~70 lambda/D degrees. Uses
render_match_scaffold.

MATCH: a 1 m dish at 10 GHz (eta~0.6) has ~38 dBi gain and a ~2 deg beam. render->match, never fit.
  python em/antenna_aperture.py
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
import sys, numpy as np
from render_match_scaffold import Benchmark, render_match
C=2.998e8
def gain_dBi(D, f, eta): lam=C/f; return 10*np.log10(eta*(np.pi*D/lam)**2)
def main():
    print("="*92); print("ANTENNA -- G=eta(pi D/lambda)^2; render->match"); print("="*92)
    def rfn(p): return gain_dBi(p.get("D",1.0), p.get("f",10e9), p["eta"])
    band=[{"eta":0.50},{"eta":0.65}]   # aperture efficiency = sigma
    res=render_match(rfn, band, {"eta":0.60},
        Benchmark("gain of a 1m dish at 10 GHz", 38.0, 1.0, "G=eta(pi D/lambda)^2", "dBi"),
        nulls=[("no aperture (D->0 -> no gain)", {"eta":0.60,"D":1e-3}, lambda g,m: g<0),
               ("isotropic limit reads ~0 dBi for tiny apertures", {"eta":0.60,"D":0.0095}, lambda g,m: abs(g)<3)],
        perturbations=[("bigger dish (D up -> more gain)", {"eta":0.60,"D":3.0}, lambda g,best: g>best)],
        notes=["a vanishing dish has no directivity (gain->0); a larger dish concentrates the beam more"])
    print(res.report())
    lam=C/10e9; bw=70*lam/1.0
    print(f"\n  diameter -> gain (10 GHz): " + "  ".join(f"{d}m->{gain_dBi(d,10e9,0.6):.0f}dBi" for d in (0.5,1,3)))
    print(f"  (4) *D^2 AND f^2: doubling the dish adds {gain_dBi(2,10e9,0.6)-gain_dBi(1,10e9,0.6):.0f} dB (=6 dB, area x4) -- and doubling frequency does the same; gain is electrical AREA")
    print(f"  (5) *NARROW BEAM: the 1 m dish at 10 GHz makes a ~{bw:.1f} deg beam (~70 lambda/D) -- high gain and a pencil beam are the same fact, why you must aim a dish")
    g4=abs((gain_dBi(2,10e9,0.6)-gain_dBi(1,10e9,0.6))-6.02)<0.1; g5=1<bw<4
    ok=res.ok and g4 and g5
    print("="*92); print("RENDER->MATCH CLOSES (antenna)" if ok else f"HONEST ok={res.ok} g4={g4} g5={g5}"); print("="*92)
    return 0 if ok else 1
if __name__=="__main__": sys.exit(main())
