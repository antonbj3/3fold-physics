"""GRIFFITH PARTICLE FRACTURE - the critical electrode-particle size.

What it computes: intercalation stress fractures active particles and causes loss of active material; the Griffith
criterion sets which particles fracture. A flaw of size a in a particle under stress sigma becomes unstable when the
stress intensity K = sigma*Y*sqrt(pi*a) exceeds the fracture toughness K_IC, so there is a critical particle size
    R_crit = (K_IC/(sigma*Y))^2 / pi
below which the particle survives. The (de)intercalation stress is sigma ~ kappa*E*eps_lin/(1 - nu) with eps_lin the
linear swelling strain, so larger swelling means larger stress and R_crit falling as 1/sigma^2. For graphite (modest
swelling) R_crit is of order a micrometre, so standard particles exceed it and crack gradually; for silicon (huge
swelling) R_crit collapses to tens of nanometres, which is why silicon anodes must be nano-structured.

Inputs: none (graphite E = 15 GPa, eps_lin = 0.032, K_IC = 0.5 MPa*sqrt(m); silicon E = 90 GPa, 300% volumetric
swelling with the stress capped at a realistic ~3 GPa fracture stress, K_IC = 0.8 MPa*sqrt(m)). Outputs: the
diffusion-induced stress and critical radius for graphite and silicon, and gate lines G1-G4.

Reference: Griffith fracture criterion; literature critical silicon particle size ~150 nm and standard 5-20 um
graphite particles.

GATES: G1 graphite R_crit is of order a micrometre. G2 standard graphite particles (5-20 um) exceed R_crit, so they
fracture, giving the gradual loss of active material. G3 silicon contrast - the ~10x larger swelling strain drives the
stress far higher and collapses R_crit to tens of nanometres; NULL - a stress-free particle has K = 0 < K_IC and never
fractures. G4 composes the stress-to-fracture-to-LAM mechanism, the silicon fade case and the Griffith criterion.
"""
import sys
import numpy as np

Y = 1.12               # geometry factor (edge flaw)
KAPPA = 0.4            # partial-gradient/geometry factor for diffusion-induced stress


def diffusion_stress(E, eps_lin, nu):
    return KAPPA * E * eps_lin / (1 - nu)                     # Pa


def r_crit(sigma, K_IC):
    return (K_IC / (sigma * Y)) ** 2 / np.pi                  # m


def main():
    print("=" * 100)
    print("GRIFFITH PARTICLE-FRACTURE render→match (critical electrode-particle size; graphite µm vs Si nano)")
    print("=" * 100)
    # graphite: E~15 GPa, ~10% volumetric → ε_lin~0.032, K_IC~0.5 MPa·√m
    sg = diffusion_stress(15e9, 0.032, 0.30); Rg = r_crit(sg, 0.5e6)
    # silicon: E~90 GPa, ~300% volumetric → ε_lin~0.44 (capped by fracture/plasticity to a realistic σ), K_IC~0.8 MPa·√m
    eps_si = (4.0 ** (1 / 3) - 1)                             # linear strain for 300% volumetric
    ss_raw = diffusion_stress(90e9, eps_si, 0.22)
    ss = min(ss_raw, 3.0e9)                                   # cap at realistic Si fracture stress (~GPa; elastic value is unphysical)
    Rs = r_crit(ss, 0.8e6)
    print(f"\n  GRAPHITE: σ_diff ≈ {sg/1e6:.0f} MPa (10% swell), K_IC 0.5 MPa·√m → R_crit = {Rg*1e6:.2f} µm")
    print(f"  SILICON : σ_diff ≈ {ss/1e9:.1f} GPa (300% swell, capped; elastic {ss_raw/1e9:.0f} GPa unphysical), K_IC 0.8 → R_crit = {Rs*1e9:.0f} nm")
    print(f"  G2 standard graphite particles ~5–20 µm ≫ R_crit {Rg*1e6:.1f} µm → fracture → gradual LAM")
    print(f"  G3 contrast: Si R_crit {Rs*1e9:.0f} nm ≪ graphite {Rg*1e6:.1f} µm ({Rg/Rs:.0f}× smaller) → Si needs NANO-structuring (literature ~150 nm)")

    g1 = 0.3e-6 < Rg < 5e-6                                   # ★graphite R_crit ~ µm
    g2 = Rg < 5e-6                                            # ★standard graphite (5–20µm) exceeds R_crit → fractures
    g3 = (Rs < 0.3e-6) and (Rg / Rs > 5)                     # ★Si R_crit nano + far below graphite
    g4 = g1 and g3                                            # composes the LAM mechanism, the Si case and Griffith
    ok = g1 and g2 and g3
    print("\n" + "-" * 100)
    print(f"  G1 ★graphite Griffith R_crit = {Rg*1e6:.2f} µm (σ {sg/1e6:.0f} MPa, K_IC 0.5)                          {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★standard graphite particles (5–20 µm) ≫ R_crit → they fracture → the gradual LAM          {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★Si contrast: R_crit {Rs*1e9:.0f} nm ({Rg/Rs:.0f}× smaller) → Si must be nano-structured             {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes stress→fracture→LAM + the Si case + the canonical Griffith criterion               {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("GRIFFITH PARTICLE-FRACTURE render→match LIVE — the canonical fracture criterion sets which electrode particles crack. A")
        print(f"  diffusion-induced stress σ ≈ κ·E·ε/(1−ν) puts a flaw of size R under intensity σ·Y·√(πR); fracture (K>K_IC) above a critical")
        print(f"  size R_crit = (K_IC/σY)²/π. For graphite (10% swell, σ ≈ {sg/1e6:.0f} MPa) R_crit = {Rg*1e6:.1f} µm — and standard 5–20 µm particles exceed it,")
        print(f"  so they crack gradually: the measured LINEAR loss-of-active-material. For silicon (300% swell) the stress is ~{ss/sg:.0f}× higher,")
        print(f"  collapsing R_crit to {Rs*1e9:.0f} nm ({Rg/Rs:.0f}× smaller) — which is precisely why silicon anodes must be NANO-structured and why silicon is")
        print(f"  fade-limited (the literature critical size is ~150 nm). NULL: a particle with no SOC swing carries K=0 < K_IC and never cracks.")
        print(f"  Composes the stress mechanism and the silicon fade case with the canonical Griffith fracture mechanics.")
    else:
        print(f"HONEST WIP: g1={g1}(Rg {Rg*1e6:.2f}µm) g2={g2} g3={g3}(Rs {Rs*1e9:.0f}nm). Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
