"""MAGNETOSTRICTION (em↔elastic coupling, a missing edge) — a ferromagnet changes shape when magnetized (Joule
magnetostriction): the strain follows the magnetization as ε=λ_s·(M/M_s)², EMERGING from minimizing the magnetoelastic +
elastic energy. ★The decisive, non-tautological signature is FREQUENCY DOUBLING: because the strain goes as M² (EVEN in M —
it does not care which way the field points), an AC drive M=M₀cos(ωt) produces strain at 2ω, NOT ω — this is exactly why a
mains transformer hums at 100/120 Hz (twice the 50/60 Hz line). GEOMETRIC/energetic: the magnetoelastic energy −b·ε·m² is
linear in strain, so its minimum against the elastic ½cε² sits at ε=(b/c)m² — strain tracks m², a parabola through the origin.

Magnetoelastic energy density E(ε)=−b·ε·m² + ½c·ε² (m=M/M_s); minimize over ε ⇒ ε=λ_s·m², λ_s=b/c (saturation magnetostriction).

FALSIFICATION (energy-min law + the 2ω signature + saturation + null): (1) ★ε(M)=λ_s(M/M_s)² emerges from argmin (∝M², EVEN);
(2) ★FREQUENCY DOUBLING — M=M₀cos(ωt) → the strain's AC power is at 2ω (FFT), the transformer-hum mechanism; (3) ★saturation
ε→λ_s as M→M_s; (4) ★NULL: b=0 (no magnetoelastic coupling) → no strain; M=0 → ε=0. Render → artifacts/magnetostriction.png.

  python em/magnetostriction.py
"""
import os
import sys
import numpy as np

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")

b = 3e-5; c = 1.0; lam_s = b / c                                      # magnetoelastic coupling, elastic modulus, λ_s


def strain_eq(m, bb=b, cc=c):
    """equilibrium strain at normalized magnetization m: argmin over ε of E(ε)=−bb·ε·m² + ½cc·ε² (done on a grid, emergent)."""
    eps = np.linspace(-2 * abs(bb) / cc - 1e-12, 2 * abs(bb) / cc + 1e-12, 4001)   # odd+symmetric ⇒ ε=0 is on the grid
    out = np.zeros_like(np.atleast_1d(m), float)
    for i, mi in enumerate(np.atleast_1d(m)):
        E = -bb * eps * mi ** 2 + 0.5 * cc * eps ** 2
        out[i] = eps[np.argmin(E)]
    return out if out.size > 1 else float(out[0])


def main():
    print("=" * 84)
    print("MAGNETOSTRICTION — strain ∝ M² from energy minimization; AC drive → 2ω (transformer hum) (em↔elastic)")
    print("=" * 84)
    print(f"\n  magnetoelastic b={b}, elastic c={c} ⇒ saturation magnetostriction λ_s=b/c={lam_s:.1e}")

    # (1) ε(M)=λ_s·m² emerges from energy minimization (∝M², even)
    m = np.linspace(-1, 1, 41)
    eps = strain_eq(m); eps_pred = lam_s * m ** 2
    e1 = np.max(np.abs(eps - eps_pred)) / lam_s; ok1 = e1 < 0.01
    even = np.max(np.abs(eps - eps[::-1])) / lam_s < 1e-9             # ε(m)=ε(−m)
    print(f"  (1) ★ε(M)=λ_s·m² from argmin: max dev {e1*100:.2f}% of λ_s; EVEN in M (ε(m)=ε(−m)): {even}  {'✓' if ok1 and even else 'FAIL'}")

    # (2) FREQUENCY DOUBLING: drive m=cos(ωt) → strain at 2ω (FFT)
    N = 4096; t = np.arange(N); f_drive = 16                          # drive frequency (cycles over the window)
    mt = np.cos(2 * np.pi * f_drive * t / N)
    eps_t = strain_eq(mt)                                             # strain follows m² → DC + 2ω
    sp = np.abs(np.fft.rfft(eps_t - eps_t.mean())); fpk = np.argmax(sp)
    ok2 = fpk == 2 * f_drive
    print(f"  (2) ★FREQUENCY DOUBLING: drive at f={f_drive} → strain AC peak at f={fpk} (=2×{f_drive}? {fpk==2*f_drive}) — the transformer hum  {'✓' if ok2 else 'FAIL'}")

    # (3) saturation: ε(M_s)=λ_s
    eps_sat = strain_eq(1.0); e3 = abs(eps_sat - lam_s) / lam_s; ok3 = e3 < 0.01
    print(f"  (3) ★saturation ε(M=M_s) = {eps_sat:.2e} vs λ_s={lam_s:.2e} (Δ{e3*100:.1f}%)  {'✓' if ok3 else 'FAIL'}")

    # (4) NULL: b=0 → no strain; M=0 → ε=0
    eps_b0 = np.max(np.abs(strain_eq(m, bb=0.0))); eps_m0 = abs(strain_eq(0.0))
    ok4 = eps_b0 < 1e-12 and eps_m0 < 1e-12
    print(f"  (4) ★NULL: b=0 → |ε|max={eps_b0:.1e}; M=0 → ε={eps_m0:.1e}  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
        ax[0].plot(m, eps / lam_s, "b-"); ax[0].set_xlabel("M/M_s"); ax[0].set_ylabel("ε/λ_s"); ax[0].set_title("ε ∝ M² (even, parabola)", fontsize=9)
        ax[1].plot(t[:512], mt[:512], "g-", lw=0.7, label="M(t)=cos(ωt)"); ax[1].plot(t[:512], eps_t[:512] / lam_s, "r-", lw=0.9, label="strain (2ω)")
        ax[1].set_xlabel("t"); ax[1].set_title("AC drive → strain at 2ω (hum)", fontsize=9); ax[1].legend(fontsize=7)
        fig.tight_layout(); os.makedirs(_ART, exist_ok=True); fig.savefig(os.path.join(_ART, "magnetostriction.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and even and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("MAGNETOSTRICTION validated — strain ∝ M² from energy minimization (em↔elastic, non-tautological 2ω signature):")
        print(f"  • the equilibrium strain tracks m² (argmin of the magnetoelastic+elastic energy), EVEN in M, saturating at λ_s;")
        print(f"  • an AC field at ω drives strain at 2ω (FFT peak at 2f) — the physical origin of the 100/120 Hz transformer hum;")
        print(f"  ⇒ em↔elastic filled (magnetostrictive sensors/actuators / sonar transducers / transformer acoustics) — the")
        print(f"    Joule effect; the Villari inverse (stress→M) is the Onsager partner, not claimed here.")
    else:
        print(f"  (1)M² {ok1 and even} (2)2ω {ok2} (3)saturation {ok3} (4)null {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
