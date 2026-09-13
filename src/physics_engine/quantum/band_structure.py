"""1D BAND STRUCTURE / PHOTONIC (PHONONIC) BANDGAP — waves in a periodic layered medium develop FORBIDDEN frequency
bands, EMERGING from the Bloch condition on the exact transfer matrix. A scalar wave (acoustic pressure / EM field) in a
bilayer period (speeds c₁,c₂, thicknesses d₁,d₂) obeys cos(K·d) = ½·Tr(M), where M is the one-period transfer matrix and
K the Bloch wavevector. Where |½·Tr(M)| ≤ 1 the wave PROPAGATES (a band); where |½·Tr(M)| > 1 the Bloch wavevector is
COMPLEX → the wave is evanescent → a BANDGAP. The gap is not coded; it falls out of the transfer matrix.

★The decisive, non-tautological anchors:
  • ★NO-CONTRAST NULL: identical layers (c₁=c₂) ⇒ |½·Tr(M)| ≤ 1 for ALL ω ⇒ NO gap, the uniform-medium linear dispersion
    ω = c·K. The periodicity-with-CONTRAST is the mechanism — without contrast, no forbidden band.
  • the first gap opens at the BRILLOUIN-ZONE EDGE K·d = π (cos(Kd) = −1) — the band folds and splits there (the Bragg
    condition), the hallmark of a periodic medium.
  • the gap WIDTH grows monotonically with the contrast (c₂/c₁ → 1 closes it) — the gap is opened BY the contrast.

Transfer matrix of a homogeneous layer (k=ω/c, thickness d) for (ψ, ψ'): [[cos kd, sin kd / k],[−k sin kd, cos kd]];
one period M = M₂·M₁ ⇒ ½Tr(M) = cos(k₁d₁)cos(k₂d₂) − ½(k₁/k₂ + k₂/k₁) sin(k₁d₁) sin(k₂d₂). FALSIFICATION: (1) gaps
emerge (|½Tr|>1 over ω ranges); (2) NULL no-contrast ⇒ no gap; (3) first gap at Kd=π; (4) gap width ∝ contrast.
Figure written to artifacts/band_structure.png.

  python quantum/band_structure.py
"""
import sys
import numpy as np
import os

_ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(_ARTIFACTS, exist_ok=True)



def half_trace(omega, c1=1.0, c2=0.5, d1=1.0, d2=1.0):
    """½·Tr of the one-period transfer matrix vs frequency (the Bloch dispersion: cos(K·d) = ½Tr)."""
    k1 = omega / c1; k2 = omega / c2
    with np.errstate(divide="ignore", invalid="ignore"):
        r = 0.5 * (k1 / k2 + k2 / k1)
    return np.cos(k1 * d1) * np.cos(k2 * d2) - r * np.sin(k1 * d1) * np.sin(k2 * d2)


def gaps(omega, ht):
    """frequency ranges where |½Tr|>1 (evanescent ⇒ bandgap)."""
    forb = np.abs(ht) > 1.0
    edges = np.where(np.diff(forb.astype(int)) != 0)[0]
    out = []
    s = None
    for i in range(len(omega)):
        if forb[i] and s is None: s = omega[i]
        if (not forb[i]) and s is not None: out.append((s, omega[i])); s = None
    if s is not None: out.append((s, omega[-1]))
    return [(a, b) for a, b in out if b - a > 1e-3]


def main():
    print("=" * 90)
    print("1D BAND STRUCTURE — forbidden bands (bandgaps) EMERGE from |½·Tr(transfer matrix)| > 1")
    print("=" * 90)
    d1 = d2 = 1.0; d = d1 + d2
    om = np.linspace(1e-3, 8.0, 20000)

    ht = half_trace(om, c1=1.0, c2=0.5)                              # contrasted bilayer (2:1 speed)
    gp = gaps(om, ht)
    htN = half_trace(om, c1=1.0, c2=1.0)                            # NULL: identical layers (no contrast)
    gpN = gaps(om, htN)

    # (3) the first gap opens at the Brillouin-zone edge Kd=π: at the gap edges |½Tr|=1; the FIRST gap should bracket a
    #     point where ½Tr = −1 (cos(Kd)=−1 ⇒ Kd=π). Check ½Tr crosses −1 inside the first gap.
    g1 = gp[0] if gp else (0, 0)
    mid = 0.5 * (g1[0] + g1[1]); ht_mid = half_trace(np.array([mid]), 1.0, 0.5)[0]
    at_bz_edge = ht_mid < -1.0                                       # inside the first gap ½Tr < −1 (Kd=π side)

    # (4) gap width vs contrast: shrink the contrast → the first gap narrows toward 0
    widths = []
    for c2 in (0.5, 0.7, 0.9, 0.99):
        gg = gaps(om, half_trace(om, 1.0, c2))
        widths.append(gg[0][1] - gg[0][0] if gg else 0.0)
    mono = all(widths[i] > widths[i + 1] for i in range(len(widths) - 1))   # narrows monotonically toward no-contrast

    g_exist = len(gp) >= 2                                           # at least the first two bandgaps emerge
    g_null = len(gpN) == 0                                           # no contrast ⇒ no gap
    g_bz = at_bz_edge
    g_width = mono and widths[0] > 0.2 and widths[-1] < 0.1
    ok = g_exist and g_null and g_bz and g_width
    print(f"\n  (1) BANDGAPS EMERGE: {len(gp)} forbidden bands in ω∈(0,8]; first = ({g1[0]:.3f},{g1[1]:.3f}) width {g1[1]-g1[0]:.3f}  {'✓' if g_exist else 'FAIL'}")
    print(f"  (2) ★NO-CONTRAST NULL (c₁=c₂): {len(gpN)} gaps — uniform medium has NONE (the linear dispersion ω=cK)  {'✓' if g_null else 'FAIL'}")
    print(f"  (3) ★FIRST GAP at Brillouin-zone edge Kd=π: ½Tr at gap centre = {ht_mid:.3f} < −1 (cos Kd=−1 ⇒ Kd=π)  {'✓' if g_bz else 'FAIL'}")
    print(f"  (4) ★GAP WIDTH ∝ CONTRAST: first-gap width for c₂/c₁=[0.5,0.7,0.9,0.99] = {[round(w,3) for w in widths]} → closes as contrast→1  {'✓' if g_width else 'FAIL'}")
    print("\n" + "=" * 90)
    if ok:
        print("VALIDATED: 1D photonic/phononic band structure — bandgaps EMERGE from the transfer matrix (non-tautological):")
        print(f"  • {len(gp)} forbidden bands fall out of |½·Tr(M)|>1 (the Bloch condition cos Kd=½Tr); they are nowhere coded.")
        print(f"  • ★NO-CONTRAST NULL: identical layers ⇒ ZERO gaps (uniform dispersion ω=cK) — the periodicity-with-contrast")
        print(f"    is the mechanism. The first gap opens at the Brillouin-zone edge Kd=π (½Tr={ht_mid:.2f}<−1), the Bragg split.")
        print(f"  • ★the gap WIDTH grows with the contrast ({widths[0]:.2f} at 2:1 → {widths[-1]:.3f} near 1:1) — opened BY the contrast.")
        print(f"  ⇒ a clean wave-in-periodic-media law (photonic/phononic crystals, Bragg mirrors, acoustic filters); EM/acoustic dual.")
    else:
        print(f"  exist {g_exist} ({len(gp)}), null {g_null} ({len(gpN)}), bz-edge {g_bz} (½Tr {ht_mid:.2f}), width {g_width} ({[round(w,2) for w in widths]}). Fix at source.")
    print("=" * 90)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        prop = np.abs(ht) <= 1.0; Kd = np.full_like(om, np.nan); Kd[prop] = np.arccos(ht[prop])
        plt.figure(figsize=(5, 4)); plt.plot(Kd / np.pi, om, ".", ms=1)
        for a, b in gp: plt.axhspan(a, b, color="red", alpha=0.12)
        plt.xlabel("K·d / π"); plt.ylabel("ω"); plt.title("band structure (red = bandgaps, emergent)")
        plt.tight_layout(); plt.savefig(os.path.join(_ARTIFACTS, "band_structure.png"), dpi=90); plt.close()
    except Exception:
        pass
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
