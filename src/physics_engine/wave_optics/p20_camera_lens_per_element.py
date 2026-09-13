"""Camera lens modelled per element: spherical and chromatic aberrations emerge from an exact Snell ray-trace through
each surface, instead of being imposed as lumped Zernike coefficients (no fit).

Finite meridional rays are traced through each spherical surface by exact vector Snell refraction, with Sellmeier
dispersion per glass (n_of from optics_achromat). The aberrations then follow from the prescription alone:
  * SPHERICAL: marginal rays of a singlet focus shorter than paraxial - the longitudinal spherical aberration (LSA)
    emerges, no coefficient assumed.
  * CHROMATIC: a per-wavelength index gives a per-wavelength focus; a singlet splits the C and F foci, an achromatic
    doublet (crown + flint, thin-lens condition phi1/V1 + phi2/V2 = 0) un-splits them.
Reproduce-before-consume: the paraxial focus (marginal height -> 0) must equal the lensmaker EFL, so the ray-trace is
calibrated against a known target - the prescription itself (radii, glasses, spacings).

INPUT: none (glass data from the Sellmeier library in optics_achromat). OUTPUT: printed gate lines; exit 0 when all
gates hold.
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
from optics_achromat import n_of, glass_nV, SELL, LD, LC, LF


def trace_ray(h, surfaces, lam):
    """exact 2D meridional ray-trace: parallel input ray at height h through spherical surfaces → axis-crossing z (back focus).
    surfaces = list of (z_vertex, R, glass_after or None for air). returns z where the exiting ray crosses y=0."""
    P = np.array([-50.0, h]); d = np.array([1.0, 0.0])           # parallel ray from the left
    n1 = 1.0
    for (zv, R, glass) in surfaces:
        n2 = 1.0 if glass is None else n_of(lam, glass)
        zc = zv + R                                              # sphere center on axis
        oc = P - np.array([zc, 0.0])
        b = 2 * np.dot(oc, d); c = np.dot(oc, oc) - R * R
        disc = b * b - 4 * c
        if disc < 0:
            return np.nan
        ts = sorted([(-b - np.sqrt(disc)) / 2, (-b + np.sqrt(disc)) / 2])
        t = next((tt for tt in ts if tt > 1e-9), None)           # first forward intersection
        if t is None:
            return np.nan
        P = P + t * d
        nrm = (P - np.array([zc, 0.0])) / R                      # outward surface normal (sign via R)
        cosi = -np.dot(d, nrm)
        if cosi < 0:
            nrm = -nrm; cosi = -np.dot(d, nrm)
        eta = n1 / n2
        k = 1 - eta * eta * (1 - cosi * cosi)
        if k < 0:
            return np.nan                                        # TIR
        d = eta * d + (eta * cosi - np.sqrt(k)) * nrm
        d = d / np.linalg.norm(d)
        n1 = n2
    return P[0] - P[1] * d[0] / d[1]                             # propagate to y=0 → axis-crossing z


def focus(surfaces, lam, h=0.5):
    return trace_ray(h, surfaces, lam)


def main():
    print("=" * 100)
    print("Per-lens-element ray-trace: spherical + chromatic aberrations EMERGE from Snell-per-surface (no fit)")
    print("=" * 100)
    # pick a crown (high Abbe V) and a flint (low V) from the Sellmeier library
    glasses = sorted(SELL.keys(), key=lambda g: glass_nV(g)[1])
    flint, crown = glasses[0], glasses[-1]
    nC = glass_nV(crown); nFl = glass_nV(flint)
    print(f"\n  glasses: crown {crown} (n_d {nC[0]:.3f}, V {nC[1]:.0f}); flint {flint} (n_d {nFl[0]:.3f}, V {nFl[1]:.0f})")

    # ── SINGLET: biconvex, lensmaker R = 2(n-1)f for f=100 mm ──
    f_target, nd = 100.0, nC[0]
    R = 2 * (nd - 1) * f_target
    t_lens = 4.0
    singlet = [(0.0, R, crown), (t_lens, -R, None)]              # R1=+R (air→crown), R2=−R (crown→air)
    bfd_par = focus(singlet, LD, h=0.3)                          # paraxial (small h)
    efl_par = bfd_par                                           # back focus ≈ EFL for a thin-ish lens from infinity (vertex≈0)
    print(f"\n  SINGLET (biconvex {crown}, R=±{R:.1f}mm, f_target={f_target}): paraxial back-focus {bfd_par:.1f} mm vs lensmaker EFL {f_target:.0f}")

    # SPHERICAL aberration: marginal vs paraxial focus
    hs = np.linspace(0.3, 15.0, 12)
    bfds = np.array([focus(singlet, LD, h=hh) for hh in hs])
    lsa = bfd_par - bfds[-1]                                     # longitudinal spherical aberration (marginal focuses shorter)
    print(f"  SPHERICAL: marginal-ray (h={hs[-1]:.0f}mm) focus {bfds[-1]:.1f} mm vs paraxial {bfd_par:.1f} ⇒ LSA {lsa:.2f} mm (emergent, undershoot)")

    # CHROMATIC: per-λ focus of the singlet (C/d/F)
    fC, fd, fF = focus(singlet, LC, 0.3), focus(singlet, LD, 0.3), focus(singlet, LF, 0.3)
    chroma_singlet = fC - fF                                     # axial chromatic focal shift
    print(f"  CHROMATIC (singlet): focus C {fC:.2f} / d {fd:.2f} / F {fF:.2f} mm ⇒ C−F split {chroma_singlet:.2f} mm")

    # ── DOUBLET achromat (crown+flint), thin-lens achromatic condition φ1/V1+φ2/V2=0, φ1+φ2=1/f ──
    V1, V2 = nC[1], nFl[1]
    phi = 1.0 / f_target
    p1 = phi * V1 / (V1 - V2); p2 = phi - p1                     # achromatic split: φ1/V1+φ2/V2=0, φ1+φ2=1/f
    inv_R1 = 0.60 * p1 / (nC[0] - 1)                            # bending DOF (chromatic is set by p1,p2 regardless of bending)
    inv_Rc = inv_R1 - p1 / (nC[0] - 1)                         # crown 2nd surface (cemented) from the crown power
    inv_R2b = inv_Rc - p2 / (nFl[0] - 1)                       # flint outer surface from the flint power
    R1d, Rc, R2b = 1 / inv_R1, 1 / inv_Rc, 1 / inv_R2b
    doublet = [(0.0, R1d, crown), (3.0, Rc, flint), (5.0, R2b, None)]
    fC2, fF2 = focus(doublet, LC, 0.3), focus(doublet, LF, 0.3)
    chroma_doublet = fC2 - fF2
    print(f"  CHROMATIC (achromat doublet {crown}+{flint}): C−F split {chroma_doublet:.2f} mm — vs singlet {chroma_singlet:.2f} (×{abs(chroma_singlet/max(abs(chroma_doublet),1e-3)):.0f} reduced)")

    g1 = abs(bfd_par - f_target) / f_target < 0.08              # paraxial focus reproduces the lensmaker EFL
    g2 = lsa > 0.5                                              # spherical aberration EMERGES (marginal undershoots, LSA finite)
    g3 = abs(chroma_singlet) > 0.5                             # chromatic EMERGES (singlet C/F split)
    g4 = abs(chroma_doublet) < 0.4 * abs(chroma_singlet)       # the multi-element achromat un-splits the chromatic
    ok = g1 and g2 and g3 and g4
    print(f"\n  (1) ★paraxial focus {bfd_par:.0f}mm reproduces the lensmaker EFL {f_target:.0f}mm — ray-trace correct  {'✓' if g1 else 'FAIL'}")
    print(f"  (2) ★SPHERICAL aberration EMERGES (LSA {lsa:.1f}mm, marginal undershoots) — from Snell, no Zernike coefficient  {'✓' if g2 else 'FAIL'}")
    print(f"  (3) ★CHROMATIC EMERGES (singlet C−F {chroma_singlet:.1f}mm) — from the Sellmeier per-λ index  {'✓' if g3 else 'FAIL'}")
    print(f"  (4) ★the multi-element ACHROMAT un-splits chromatic ({chroma_doublet:.2f} vs {chroma_singlet:.1f}mm) — correction emerges from the prescription  {'✓' if g4 else 'FAIL'}")
    print("\n" + "=" * 100)
    if ok:
        print("Per-lens-element ray-trace render→matched, no fit (aberrations emerge from the real lens design):")
        print(f"  • a finite Snell-per-surface meridional ray-trace (Sellmeier dispersion per glass) replaces the lumped-Zernike pupil: the")
        print(f"    paraxial focus reproduces the lensmaker EFL ({bfd_par:.0f}≈{f_target:.0f}mm), and the SPHERICAL (LSA {lsa:.1f}mm) + CHROMATIC (C−F {chroma_singlet:.1f}mm)")
        print(f"    aberrations EMERGE from the prescription — no assumed Zernike coefficients.")
        print(f"  • ★the MULTI-ELEMENT achromat (crown {crown} V{nC[1]:.0f} + flint {flint} V{nFl[1]:.0f}) un-splits the chromatic to {chroma_doublet:.2f}mm (×{abs(chroma_singlet/max(abs(chroma_doublet),1e-3)):.0f}")
        print(f"    smaller) — the correction EMERGES from the two glasses + the element spacing, exactly as a real objective works. The camera")
        print(f"    camera lens is a physical prescription, differentiable, calibrated against the prescription (no fit).")
    else:
        print(f"  HONEST: EFL {bfd_par:.0f}/{f_target:.0f}, LSA {lsa:.2f}, chroma singlet {chroma_singlet:.2f} / doublet {chroma_doublet:.2f}. Inspect.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
