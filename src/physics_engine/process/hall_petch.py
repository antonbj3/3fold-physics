"""HALL-PETCH GRAIN-SIZE STRENGTHENING — RENDER→MATCH why finer-grained metal is stronger, the materials half of the
microstructure story (pairs with dendrite_arm_spacing: rapid solidification → fine grains → THIS makes them strong). Grain
boundaries block dislocation motion: a dislocation piles up at a boundary, and the stress to push slip into the next grain
rises as the pile-up shortens — giving the yield strength
    σ_y = σ_0 + k_HP / √d
with σ_0 the lattice friction stress, k_HP the Hall-Petch slope, d the grain diameter. Halving the grain raises the
boundary contribution by √2.

I/O: no input files; prints the rendered yield strength, the gate lines and a PASS/FAIL verdict (exit 0 on pass).
MATCH: mild steel at d≈10 µm yields ≈ 250–290 MPa; σ_y−σ_0 ∝ d^{−1/2}. render→match, never fit: d is the microstructure;
σ_0 and k_HP are the material (the σ).
Reference: Hall (1951) / Petch (1953) grain-size strengthening law; ferrite σ_0≈70 MPa, k_HP≈0.6 MPa·m^0.5.
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


def sigma_y(s0, k, d):
    return s0 + k / d ** 0.5                                         # MPa (k in MPa·m^0.5, d in m)


def main():
    print("=" * 92)
    print("HALL-PETCH — σ_y = σ_0 + k/√d; finer grains → stronger metal (√d law); render→match")
    print("=" * 92)
    d0 = 10e-6
    def rfn(p):
        return sigma_y(p["s0"], p["k"], p.get("d", d0))
    band = [{"s0": 60.0, "k": 0.55}, {"s0": 80.0, "k": 0.65}]       # lattice friction × HP slope (material) = σ
    res = render_match(
        rfn, band, {"s0": 70.0, "k": 0.60},
        Benchmark("mild-steel yield (d≈10 µm)", 270.0, 30.0, "Hall-Petch; ferrite ~250-290 MPa", "MPa"),
        nulls=[("single crystal (d→large)", {"s0": 70.0, "k": 0.60, "d": 1.0}, lambda y, m: abs(y - 70) < m / 3)],
        perturbations=[("finer grains (d↓)", {"s0": 70.0, "k": 0.60, "d": 1e-6}, lambda y, best: y > best)],
        notes=["d→∞ (single crystal): only the lattice friction σ_0 remains; finer grains add k/√d strength"])
    print(res.report())
    # sqrt(d) scaling of the boundary contribution + the grain-size range
    ds = np.array([1e-6, 4e-6, 16e-6, 64e-6]); boundary = np.array([sigma_y(70.0, 0.60, d) - 70.0 for d in ds])
    slope = np.polyfit(np.log(ds), np.log(boundary), 1)[0]
    coarse, fine = sigma_y(70.0, 0.60, 100e-6), sigma_y(70.0, 0.60, 1e-6)
    print(f"\n  grain size → yield (σ_0=70, k=0.6): " + "  ".join(f"{d*1e6:.0f}µm→{sigma_y(70.0,0.60,d):.0f}" for d in ds) + " MPa")
    print(f"  (4) ★√d SCALING: boundary term (σ_y−σ_0) ∝ d^n, slope {slope:.3f} (predicted −0.5)")
    print(f"  (5) ★RANGE: coarse 100 µm → {coarse:.0f} MPa vs fine AM 1 µm → {fine:.0f} MPa — a {fine/coarse:.1f}× yield gain from grain refinement")
    g4 = abs(slope + 0.5) < 0.005
    g5 = fine > 2 * coarse
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (Hall-Petch) — grain boundaries set the strength, predicted not fitted:")
        print(f"  • σ_y=σ_0+k/√d renders {res.best:.0f} MPa for 10 µm steel (band [{res.band_lo:.0f},{res.band_hi:.0f}]=material-σ); measured ~270 sits {res.bounds}.")
        print(f"  • the boundary contribution scales as d^{{−1/2}} (slope {slope:.3f}); refining 100 µm→1 µm raises yield {fine/coarse:.1f}× —")
        print(f"    so the dendrite cell's fine AM/rapid-solidification microstructure is exactly why AM parts are strong. Forming, AM, weld HAZ.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, √d-slope {slope:.3f}, range {coarse:.0f}/{fine:.0f}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
