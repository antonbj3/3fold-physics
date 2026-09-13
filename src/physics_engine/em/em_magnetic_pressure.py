"""EM↔ELASTIC MAGNETIC PRESSURE — RENDER→MATCH why the strongest steady magnets top out near ~45 T no matter the current: a
magnetic field stores energy density B²/2μ₀, which acts as a real outward PRESSURE on the windings (Maxwell stress). At high
field that pressure rivals the strength of steel, so the field ceiling is STRUCTURAL, not electrical:
    P_mag = B²/(2μ₀)   →   the coil bursts when P_mag ≳ σ_yield   →   B_max = √(2μ₀·σ_yield)
Uses render_match_scaffold. The em↔elastic edge (MRI, fusion TF coils, maglev, railguns,
EM forming).

MATCH: the record steady field is ~45 T (NHMFL hybrid); √(2μ₀·σ) for real structural strengths (0.8–1.5 GPa) brackets it,
and the engine UPPER-BOUNDS what stronger materials could reach. The magnetic pressure at known magnets must also land: MRI
1.5 T → ~9 bar, fusion 12 T → ~57 MPa. render→match, never fit: μ₀ is fundamental, σ_yield is the material (= the band).

  python em/em_magnetic_pressure.py
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

MU0 = 4 * np.pi * 1e-7


def P_mag(B):
    return B ** 2 / (2 * MU0)                                        # Pa


def B_max(sigma_yield):
    return (2 * MU0 * sigma_yield) ** 0.5                            # T, structural field ceiling


def main():
    print("=" * 92)
    print("EM↔ELASTIC MAGNETIC PRESSURE — B²/2μ₀ bursts the coil; the record-field ceiling is STRUCTURAL; render→match")
    print("=" * 92)
    def rfn(p):
        return B_max(p["sig"])                                       # T
    band = [{"sig": 0.8e9}, {"sig": 1.5e9}]                          # effective structural strength (steel→composite) = σ
    res = render_match(
        rfn, band, {"sig": 1.1e9},
        Benchmark("record steady magnetic field", 45.5, 3.0, "NHMFL 45 T hybrid; structurally limited", "T"),
        nulls=[("no material strength (σ→0)", {"sig": 1e3}, lambda b, m: b < m / 50)],
        perturbations=[("stronger structure (σ↑)", {"sig": 3e9}, lambda b, best: b > best)],
        notes=["σ→0: no coil can hold any field (B_max→0); a stronger jacket lets the field go higher (B∝√σ)"])
    print(res.report())
    # ★magnetic pressure at known magnets (the Maxwell stress, geometric in B²)
    print("\n  magnetic pressure P=B²/2μ₀ at real magnets:")
    for name, B in (("MRI 1.5 T", 1.5), ("maglev 1 T", 1.0), ("LHC dipole 8.3 T", 8.3), ("ITER TF 11.8 T", 11.8), ("HTS 20 T", 20.0)):
        P = P_mag(B); tag = " ≳ steel yield → needs structure" if P > 250e6 else ""
        print(f"    {name:18s}: {P/1e6:7.1f} MPa  ({P/1e5:6.0f} bar){tag}")
    # ★B² scaling
    Bs = np.array([2.0, 4.0, 8.0, 16.0]); Ps = np.array([P_mag(b) for b in Bs])
    slope = np.polyfit(np.log(Bs), np.log(Ps), 1)[0]
    P_iter = P_mag(11.8) / 1e6
    print(f"\n  (4) ★B² SCALING: log-log slope {slope:.2f} (predicted 2.0); ITER 11.8 T → {P_iter:.0f} MPa (documented TF structural design)")
    print(f"  (5) ★STRUCTURAL CEILING: B_max=√(2μ₀σ) renders {res.band_lo:.0f}–{res.band_hi:.0f} T (σ=0.8–1.5 GPa); record 45 T sits {res.bounds}")
    g4 = abs(slope - 2.0) < 0.02 and 50 < P_iter < 65
    ok = res.ok and g4
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (magnetic pressure) — the field ceiling is a strength-of-materials limit, predicted not fitted:")
        print(f"  • B²/2μ₀ is a real pressure: 11.8 T fusion coils carry {P_iter:.0f} MPa, a 20 T HTS magnet {P_mag(20)/1e6:.0f} MPa (past steel's yield → reinforced).")
        print(f"  • the burst condition B_max=√(2μ₀σ) renders {res.band_lo:.0f}–{res.band_hi:.0f} T for real structural strengths; the ~45 T record sits at the")
        print(f"    {res.bounds} edge — stronger jackets are the only way up (B∝√σ). The engine bounds the record and names the limiter: materials, not current.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, B²-slope {slope:.2f}, ITER P {P_iter:.0f} MPa. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
