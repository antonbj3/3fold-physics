"""KELVIN-HELMHOLTZ — the unstable wavelength is the gravity-capillary balance, the 5th instability class (consumes kelvin_helmholtz_onset).

A 5th distinct instability class in the taxonomy (after fold/Hopf/chaos/morphological): the Kelvin-Helmholtz SHEAR
instability — wind raising waves on water. Geometrically, a small crest narrows the faster stream above it, so by Bernoulli the
pressure there drops and the crest grows; gravity (penalising long waves) and surface tension (penalising short waves) restore.
The wave that destabilises first is the SLOWEST one — the minimum-phase-speed wave, where the two restoring effects balance,
at k* = √(gΔρ/σ), i.e. λ* = the min-phase-speed wavelength ~1.7 cm. Above a critical shear ΔU_crit ~ 6.6 m/s the wind overtakes it and
the interface rolls up. This is the SAME structure as the Mullins-Sekerka morphology: the selected wavelength is the balance of
two competing scales — here gravity vs capillarity, there diffusion vs capillarity. kelvin_helmholtz_onset owns the forward (imported, never edited); this module cross-validates the
wavelength against its k_star and maps the new class.

GATES (null/control each — every one can fail):
 (G0) CONSUME THE KH FORWARD — U_crit / k_star imported.
 (G1) THE ONSET — ΔU_crit for air-over-water ≈ 6.6 m/s (the classic minimum wind to raise waves); stronger surface tension stiffens the interface (σ↑ → U_crit↑); NULL: no gravity AND no surface tension → any shear is unstable (U_crit → 0).
 (G2) WAVELENGTH = MINIMUM-PHASE-SPEED WAVE — my dispersion c²(k)=gΔρ/(kρ_s)+σk/ρ_s has its minimum exactly at the imported k_star (cross-method), giving λ* = 2π/k* ≈ 17 mm (the min-phase-speed wavelength).
 (G3) GRAVITY-CAPILLARY BALANCE (geometric; unifies with MS) — at k* the gravity restoring term equals the capillary one (the two competing scales balance); same wavelength-selection structure as the Mullins-Sekerka morphology. NULL: σ→0 removes the capillary branch → no interior minimum (no selected wavelength).

python thermal/spine_kelvin_helmholtz_wavelength.py
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
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
try:
    from kelvin_helmholtz_onset import U_crit, k_star, SIGMA, G, RHO_A, RHO_W
except Exception as e:
    print(f"VERDICT: cannot consume the KH forward ({e}); KH wavelength GATED. (honest-negative)"); sys.exit(1)

RS = RHO_A + RHO_W; DR = RHO_W - RHO_A


def c2(k, sigma=SIGMA):  return G * DR / (k * RS) + sigma * k / RS     # interfacial wave phase speed² (gravity + capillary)


def main():
    print("=" * 98)
    print("KELVIN-HELMHOLTZ — the unstable wavelength is the gravity-capillary balance (5th instability class)")
    print("=" * 98)
    g0 = U_crit() > 0 and k_star() > 0
    print(f"\n(G0) CONSUME THE KH FORWARD — U_crit={U_crit():.2f} m/s, k_star={k_star():.0f} /m: {'PASS' if g0 else 'FAIL'}")

    uc, uc_stiff, uc_null = U_crit(), U_crit(sigma=0.30), U_crit(sigma=1e-9, g=1e-9)
    g1 = 5.5 < uc < 7.5 and uc_stiff > uc and uc_null < 0.05
    print(f"\n(G1) THE ONSET — ΔU_crit={uc:.2f} m/s (≈6.6, classic wind-on-water); σ↑→U_crit {uc_stiff:.1f}>{uc:.1f}; NULL no-g-no-σ → U_crit={uc_null:.3f} (any shear unstable): {'PASS' if g1 else 'FAIL'}")

    kk = np.linspace(50, 1500, 6000); k_min = float(kk[np.argmin(c2(kk))])
    lam_mm = 2 * np.pi / k_star() * 1000
    g2 = abs(k_min - k_star()) / k_star() < 0.02 and 14 < lam_mm < 20
    print(f"\n(G2) WAVELENGTH = MIN-PHASE-SPEED WAVE — my dispersion argmin k={k_min:.0f}/m = the imported k_star={k_star():.0f}/m (cross-method); λ*={lam_mm:.1f} mm (min-phase-speed wavelength): {'PASS' if g2 else 'FAIL'}")

    ks = k_star(); grav_term = G * DR / (ks * RS); cap_term = SIGMA * ks / RS
    kk2 = np.linspace(50, 1500, 6000); has_min_sigma = 50 < kk2[np.argmin(c2(kk2))] < 1500
    has_min_null = 50 < kk2[np.argmin(c2(kk2, sigma=1e-9))] < 1499                       # σ→0: argmin runs to the edge (no interior min)
    g3 = abs(grav_term - cap_term) / grav_term < 0.02 and has_min_sigma and not has_min_null
    print(f"\n(G3) GRAVITY-CAPILLARY BALANCE — at k* gravity term {grav_term:.3f} = capillary term {cap_term:.3f} (the two scales balance, like MS); NULL σ→0 no interior min={not has_min_null}: {'PASS' if g3 else 'FAIL'}")

    allok = g0 and g1 and g2 and g3
    print("\n" + "=" * 98)
    if allok:
        print("VERDICT: the Kelvin-Helmholtz interface destabilises first at its SLOWEST wave — the minimum-phase-speed mode where gravity")
        print(f"  (∝1/k) and surface tension (∝k) restoring exactly balance, k*=√(gΔρ/σ), λ*={lam_mm:.0f} mm (the min-phase-speed wavelength), above ΔU_crit={uc:.1f} m/s. My")
        print(f"  dispersion minimum reproduces the imported k_star to <2% (cross-method), and the balance-of-two-competing-scales wavelength selection is the SAME")
        print(f"  structure as the Mullins-Sekerka morphology (gravity↔capillarity here, diffusion↔capillarity there). 5th instability class on the imported forward.")
    else:
        print(f"VERDICT: NOT all pass — G0 {g0} G1 {g1} G2 {g2} G3 {g3}. Fix at SOURCE.")
    print("=" * 98)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
