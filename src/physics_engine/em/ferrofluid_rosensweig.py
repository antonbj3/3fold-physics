"""ROSENSWEIG / NORMAL-FIELD INSTABILITY (phase↔em coupling, a missing edge) — a pool of ferrofluid in a vertical magnetic
field spontaneously erupts into a regular array of SPIKES once the field passes a threshold. ★The mechanism is an energy
competition on the free surface: the magnetic field CONCENTRATES at peaks (lowering magnetic energy — destabilizing ∝M²k),
while surface tension (∝γk²) and gravity (∝ρg) resist. ★The non-tautological GEOMETRIC result: at onset the spike spacing is
the CAPILLARY LENGTH, k_c=√(ρg/γ) — it EMERGES from minimizing the stabilizing (ρg/k+γk), independent of the magnetic
coupling; below the critical field the flat surface is stable to ALL wavenumbers, above it a band goes unstable.

Marginal-stability function D(k)=ρg+γk²−C_mag·M²·k (D<0 ⇒ that mode grows). Onset: min_k D = 0.

FALSIFICATION (the capillary wavenumber + onset structure + null): (1) ★at onset the marginal wavenumber k_c=√(ρg/γ) (emerges
from the dispersion's marginal point); (2) ★the spike spacing λ_c=2π√(γ/ρg) (the capillary length); (3) ★critical field —
M<M_c: D>0 ∀k (flat stable); M>M_c: a band has D<0 (spikes); M_c²=2√(γρg)/C_mag; (4) ★NULL: M=0 → D=ρg+γk²>0 ∀k (always
flat). Render → artifacts/rosensweig.png.

  python em/ferrofluid_rosensweig.py
"""
import os
import sys
import numpy as np

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

rho = 1200.0; g = 9.81; gamma = 0.025; C_mag = 1.0                    # ferrofluid density, g, surface tension, magnetic coupling


def D(k, M):
    """marginal-stability function: D<0 ⇒ the surface mode of wavenumber k grows (spikes)."""
    return rho * g + gamma * k ** 2 - C_mag * M ** 2 * k


def min_over_k(M, kmax=4000.0, n=200000):
    k = np.linspace(1.0, kmax, n); d = D(k, M); i = int(np.argmin(d))
    return d[i], k[i]


def main():
    print("=" * 84)
    print("ROSENSWEIG / NORMAL-FIELD INSTABILITY — ferrofluid spikes; spacing = capillary length (phase↔em)")
    print("=" * 84)
    k_cap = np.sqrt(rho * g / gamma); lam_cap = 2 * np.pi / k_cap
    M_c = np.sqrt(2 * np.sqrt(gamma * rho * g) / C_mag)              # critical magnetization (min_k D = 0)
    print(f"\n  ρ={rho}, γ={gamma} N/m ⇒ capillary k_c=√(ρg/γ)={k_cap:.1f} /m, λ_c={lam_cap*1e3:.2f} mm; critical M_c={M_c:.1f}")

    # (1) at onset (M=M_c) the marginal wavenumber = k_c = √(ρg/γ)
    d_onset, k_onset = min_over_k(M_c)
    e1 = abs(k_onset - k_cap) / k_cap; ok1 = e1 < 0.01
    print(f"  (1) ★onset marginal wavenumber k={k_onset:.1f} /m vs √(ρg/γ)={k_cap:.1f} (Δ{e1*100:.1f}%) — the capillary length emerges  {'✓' if ok1 else 'FAIL'}")

    # (2) spike spacing λ_c = 2π√(γ/ρg)
    lam_meas = 2 * np.pi / k_onset; e2 = abs(lam_meas - lam_cap) / lam_cap; ok2 = e2 < 0.01
    print(f"  (2) ★spike spacing λ_c = {lam_meas*1e3:.2f} mm vs 2π√(γ/ρg)={lam_cap*1e3:.2f} (Δ{e2*100:.1f}%)  {'✓' if ok2 else 'FAIL'}")

    # (3) critical field: below M_c flat-stable (min D>0), above unstable (min D<0); onset min D≈0
    d_below, _ = min_over_k(0.9 * M_c); d_above, _ = min_over_k(1.1 * M_c)
    ok3 = d_below > 0 and d_above < 0 and abs(d_onset) / (rho * g) < 0.01
    print(f"  (3) ★critical field M_c: min_k D = {d_below:.1f}(0.9M_c, stable) → {d_onset:.2f}(M_c, marginal) → {d_above:.1f}(1.1M_c, SPIKES)  {'✓' if ok3 else 'FAIL'}")

    # (4) NULL: M=0 → D>0 for all k (flat always stable)
    d_null, _ = min_over_k(0.0); ok4 = d_null > 0
    print(f"  (4) ★NULL M=0: min_k D = {d_null:.1f} > 0 (no field → flat surface stable to all modes)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.8), dpi=120)
        k = np.linspace(1, 2.5 * k_cap, 600)
        for M, lab in [(0.0, "M=0"), (0.9 * M_c, "0.9 M_c"), (M_c, "M_c"), (1.1 * M_c, "1.1 M_c (spikes)")]:
            ax.plot(k, D(k, M), lw=0.9, label=lab)
        ax.axhline(0, color="k", lw=0.6); ax.axvline(k_cap, color="grey", ls=":", lw=0.8, label="k_c=√(ρg/γ)")
        ax.set_xlabel("wavenumber k (1/m)"); ax.set_ylabel("D(k) (D<0 ⇒ unstable)"); ax.set_title("Rosensweig: instability at the capillary k_c", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); os.makedirs(_ART, exist_ok=True); fig.savefig(os.path.join(_ART, "rosensweig.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("ROSENSWEIG INSTABILITY validated — ferrofluid spikes at the capillary scale (phase↔em, the geometric onset):")
        print(f"  • the surface goes unstable above M_c with spacing = the capillary length λ_c={lam_cap*1e3:.1f} mm (k_c=√(ρg/γ)),")
        print(f"    emerging from the surface-tension/gravity minimum — independent of the magnetic coupling; below M_c it is flat;")
        print(f"  ⇒ phase↔em filled (ferrofluid spikes / magnetic-field sensors / soft-robot actuators / heat-transfer surfaces) —")
        print(f"    the Kelvin body force on a magnetized free surface.")
    else:
        print(f"  (1)k_c {ok1} (2)λ_c {ok2} (3)M_c {ok3} (4)null {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
