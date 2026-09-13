"""BRAGG X-RAY DIFFRACTION — RENDER→MATCH how you read a crystal's atomic spacing off a diffractometer, and why it takes
X-rays to do it. Planes of atoms spaced d reflect X-rays; the reflections from successive planes add in phase only when the
extra path 2d·sinθ is a whole number of wavelengths:
    n·λ = 2d·sinθ
so each set of lattice planes gives a sharp peak at a known angle, and the peak angle MEASURES the spacing d. The wavelength
must be ≲ 2d (≈ Å) — which is why crystallography uses X-rays, not visible light. Uses render_match_scaffold (a
clean materials-characterization primitive — distinct from the optical grating / electronic-band cells).

MATCH: rock-salt NaCl (a=5.64 Å) with Cu-Kα (1.5406 Å) puts its (200) reflection at 2θ≈31.7°, the (220) at ≈45.5°, the (222)
at ≈56.5°. render→match, never fit: λ and the lattice constant are physical; only the lattice-constant tolerance is the σ.
Reference: powder XRD of rock-salt NaCl with Cu-Kα (Bragg's law, nλ=2d·sinθ).
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

CU_KA = 1.5406                                                      # Cu-Kα wavelength [Å]
A_NACL = 5.6402                                                     # NaCl lattice constant [Å]


def two_theta(lam, d, n=1):
    s = n * lam / (2 * d)
    return 2 * np.degrees(np.arcsin(s)) if s <= 1.0 else 999.0     # 999 = no Bragg peak (λ too long)


def d_hkl(a, h, k, l):
    return a / np.sqrt(h ** 2 + k ** 2 + l ** 2)                    # cubic d-spacing


def main():
    print("=" * 92)
    print("BRAGG DIFFRACTION — nλ=2d·sinθ; NaCl powder pattern with Cu-Kα; render→match")
    print("=" * 92)
    d200 = d_hkl(A_NACL, 2, 0, 0)
    def rfn(p):
        return two_theta(p.get("lam", CU_KA), d_hkl(p["a"], 2, 0, 0))
    band = [{"a": 5.61}, {"a": 5.67}]                              # lattice constant tolerance (sample/thermal) = σ
    res = render_match(
        rfn, band, {"a": A_NACL},
        Benchmark("NaCl (200) reflection 2θ (Cu-Kα)", 31.7, 0.4, "powder XRD; NaCl (200)", "deg"),
        nulls=[("visible light (λ=500 nm ≫ 2d)", {"a": A_NACL, "lam": 5000.0}, lambda t, m: t > 180)],
        perturbations=[("longer wavelength (Cr-Kα 2.29 Å)", {"a": A_NACL, "lam": 2.29}, lambda t, best: t > best)],
        notes=["visible light has λ ≫ 2d → no real Bragg angle (no diffraction); a longer X-ray wavelength pushes peaks to higher 2θ"])
    print(res.report())
    # ★the multi-plane powder pattern + the d-spacing readout + the X-ray necessity
    planes = [(2, 0, 0), (2, 2, 0), (2, 2, 2), (4, 0, 0)]
    print("\n  NaCl powder pattern (Cu-Kα):")
    for hkl in planes:
        d = d_hkl(A_NACL, *hkl); tt = two_theta(CU_KA, d)
        print(f"    ({hkl[0]}{hkl[1]}{hkl[2]}): d={d:.3f} Å  →  2θ={tt:.1f}°")
    d_readout = CU_KA / (2 * np.sin(np.radians(31.7 / 2)))         # invert the measured peak → d
    ok_vis = two_theta(500 * 10, d200) > 180                       # 500 nm = 5000 Å, no peak
    print(f"  (4) ★d READOUT: inverting the measured 31.7° peak gives d={d_readout:.3f} Å — the (200) spacing, i.e. a={2*d_readout:.2f} Å; the diffractometer measures the lattice")
    print(f"  (5) ★X-RAYS NEEDED: at λ=500 nm (visible) 2d/λ={2*d200/5000:.1e} ≪ 1 → no diffraction; only λ≲2d≈Å (X-rays) reflects — why crystallography is X-ray")
    g4 = abs(d_readout - d200) / d200 < 0.02
    g5 = ok_vis and two_theta(CU_KA, d_hkl(A_NACL, 2, 2, 0)) > two_theta(CU_KA, d200)   # higher-index planes at higher 2θ
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (Bragg diffraction) — lattice spacing from peak angle, predicted not fitted:")
        print(f"  • nλ=2d·sinθ renders the NaCl (200) at {res.best:.1f}° (band [{res.band_lo:.1f},{res.band_hi:.1f}]=lattice σ); powder XRD reads ~31.7°.")
        print(f"  • inverting the angle returns d={d_readout:.3f} Å (a={2*d_readout:.2f} Å) — the peak IS a ruler for the atomic spacing, across every plane.")
        print(f"  • and only X-rays (λ≈Å≲2d) diffract — visible light can't resolve atoms. The basis of every phase-ID / strain / texture measurement.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, d-readout {g4}, ordering/X-ray {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
