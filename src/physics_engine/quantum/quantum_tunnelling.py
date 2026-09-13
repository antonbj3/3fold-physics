"""QUANTUM TUNNELLING / BARRIER SCATTERING — the transmission coefficient T(E) EMERGES from matching the wavefunction
across a potential barrier (transfer matrix), and reproduces the two signatures that have no classical analogue: a particle
with E<V₀ STILL gets through (T>0, exponentially suppressed), and a particle with E>V₀ is sometimes perfectly transmitted
(resonances, T=1). ★The transfer matrix only encodes wave-matching at the interfaces — T comes out of the SOLVE, and the
exponential decay rate and the resonance positions are physical predictions, not coded constants (the emergence lesson).

This adds a SCATTERING capability to the quantum tier (the bound-state spectrum lives in schrodinger_1d.py). ℏ=m=1.
A plane wave e^{ikx} hits a piecewise-constant potential; in each region ψ=Ae^{iqx}+Be^{-iqx}; continuity of ψ,ψ' at each
interface gives a 2×2 transfer matrix; T=|t|² from the overall matrix.

FALSIFICATION (independent closed form + physical predictions): (1) ★T(E) from the transfer matrix matches the analytic
rectangular-barrier formula; (2) ★TUNNELLING E<V₀: T>0 and decays exponentially in barrier width a (ln T linear, slope
−2κ with κ=√(2(V₀−E))); (3) ★RESONANT transmission E>V₀: T=1 exactly at ka=nπ (the Ramsauer-Townsend resonances); (4)
NULL/limits: V₀→0 ⇒ T→1, very thick barrier ⇒ T→0. Figure written to artifacts/quantum_tunnelling.png.

  python quantum/quantum_tunnelling.py
"""
import sys
import numpy as np
import os

_ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(_ARTIFACTS, exist_ok=True)



def transmission(E, V0, a, regions=None):
    """T for a 1D piecewise-constant barrier of height V0, width a, via transfer matrix. E may be array."""
    E = np.atleast_1d(E).astype(float)
    k0 = np.sqrt(2 * E + 0j)                      # outside (V=0)
    q = np.sqrt(2 * (E - V0) + 0j)               # inside the barrier
    T = np.zeros(len(E))
    for i in range(len(E)):
        T[i] = _T_single(k0[i], q[i], a)
    return T if T.size > 1 else float(T[0])


def _interface(k_from, k_to):
    """transfer matrix across a step from wavevector k_from to k_to (match ψ, ψ')."""
    return 0.5 * np.array([[1 + k_from / k_to, 1 - k_from / k_to],
                           [1 - k_from / k_to, 1 + k_from / k_to]])


def _prop(k, d):
    """free propagation by distance d at wavevector k."""
    return np.array([[np.exp(-1j * k * d), 0], [0, np.exp(1j * k * d)]])


def _T_single(k0, q, a):
    M = _interface(k0, q) @ _prop(q, a) @ _interface(q, k0)
    t = 1.0 / M[0, 0]                            # transmission amplitude
    return float(np.abs(t) ** 2)


def T_analytic(E, V0, a):
    """closed-form rectangular barrier transmission (E<V0 sinh branch, E>V0 sin branch)."""
    E = np.atleast_1d(E).astype(float); out = np.zeros(len(E))
    for i, e in enumerate(E):
        if abs(e - V0) < 1e-9:
            kappa = np.sqrt(2 * V0); out[i] = 1.0 / (1 + (kappa * a) ** 2 / 2); continue
        if e < V0:
            kappa = np.sqrt(2 * (V0 - e)); out[i] = 1.0 / (1 + V0 ** 2 * np.sinh(kappa * a) ** 2 / (4 * e * (V0 - e)))
        else:
            kk = np.sqrt(2 * (e - V0)); out[i] = 1.0 / (1 + V0 ** 2 * np.sin(kk * a) ** 2 / (4 * e * (e - V0)))
    return out if out.size > 1 else float(out[0])


def main():
    print("=" * 84)
    print("QUANTUM TUNNELLING — T(E) EMERGES from wave-matching across a barrier (quantum scattering tier)")
    print("=" * 84)
    V0, a = 4.0, 1.0

    # (1) transfer-matrix T vs analytic
    E = np.linspace(0.05, 12.0, 400)
    Tn = transmission(E, V0, a); Ta = T_analytic(E, V0, a)
    e1 = np.max(np.abs(Tn - Ta)); ok1 = e1 < 1e-6
    print(f"\n  barrier V₀={V0}, width a={a}")
    print(f"  (1) ★transfer-matrix T(E) vs analytic rectangular-barrier formula: max Δ{e1:.1e}  {'✓' if ok1 else 'FAIL'}")

    # (2) ★tunnelling: E<V0, ln T linear in barrier width, slope -2κ
    Etun = 1.0; kappa = np.sqrt(2 * (V0 - Etun))
    widths = np.linspace(0.6, 2.0, 8)
    Tw = np.array([transmission(Etun, V0, w) for w in widths])
    slope, b = np.polyfit(widths, np.log(Tw), 1)
    e2 = abs(slope - (-2 * kappa)) / (2 * kappa); ok2 = e2 < 0.05 and np.all(Tw > 0) and np.all(Tw < 1)
    print(f"  (2) ★TUNNELLING (E={Etun}<V₀): T>0 below the barrier, ln T linear in width — slope {slope:.3f} vs −2κ={-2*kappa:.3f} (Δ{e2*100:.1f}%)  {'✓' if ok2 else 'FAIL'}")
    print(f"      T(width) = {np.round(Tw,4)} for a={np.round(widths,2)} (exponential suppression, classically forbidden)")

    # (3) ★resonance STRUCTURE by INDEPENDENT peak detection (non-tautological — falsifies a wrong V₀/a, which evaluating T
    #     at the formula-predicted E_res cannot; audit-corrected from the sin(nπ)=0 identity). Find the peaks on a fine T(E)
    #     sweep by argmax, then verify their MEASURED energies match V₀+(nπ/a)²/2, AND the inter-resonance dips are real (<1).
    Es = np.linspace(V0 + 0.05, V0 + (3.3 * np.pi / a) ** 2 / 2, 5000)
    Ts = transmission(Es, V0, a)
    loc = 1 + np.where((Ts[1:-1] > Ts[2:]) & (Ts[1:-1] >= Ts[:-2]) & (Ts[1:-1] > 0.999))[0]   # local maxima ≈1 = resonances
    E_peaks = Es[loc][:3]
    n_res = np.array([1, 2, 3]); E_pred = V0 + (n_res * np.pi / a) ** 2 / 2
    pos_err = np.max(np.abs(E_peaks - E_pred)) / V0 if len(E_peaks) >= 3 else 9.9    # MEASURED peak positions vs prediction
    T_dip = float(transmission(V0 + (0.5 * np.pi / a) ** 2 / 2, V0, a))              # first (deepest) anti-resonance dip
    ok3 = len(E_peaks) >= 3 and pos_err < 0.02 and T_dip < 0.8
    print(f"  (3) ★RESONANCE STRUCTURE (E>V₀): MEASURED peak energies {np.round(E_peaks,2)} vs predicted V₀+(nπ/a)²/2 {np.round(E_pred,2)} (Δ{pos_err*100:.1f}%); dip T(ka=π/2)={T_dip:.3f}<1 — positions emerge, oscillates  {'✓' if ok3 else 'FAIL'}")

    # (4) limits
    T_noBarrier = transmission(2.0, 1e-9, a); T_thick = transmission(1.0, V0, 8.0)
    ok4 = abs(T_noBarrier - 1.0) < 1e-6 and T_thick < 1e-6
    print(f"  (4) LIMITS: V₀→0 ⇒ T={T_noBarrier:.6f} (→1); thick barrier (a=8) ⇒ T={T_thick:.2e} (→0)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=110)
        ax[0].plot(E, Tn, "b-", lw=0.9); ax[0].axvline(V0, color="k", ls=":", lw=0.8, label=f"V₀={V0}")
        for er in E_peaks: ax[0].plot(er, 1.0, "r^", ms=6)
        ax[0].set_xlabel("E"); ax[0].set_ylabel("T"); ax[0].set_title("transmission: tunnelling (E<V₀) + resonances (E>V₀)", fontsize=8); ax[0].legend(fontsize=7)
        ax[1].semilogy(widths, Tw, "go-", ms=4); ax[1].set_xlabel("barrier width a"); ax[1].set_ylabel("T (log)"); ax[1].set_title(f"tunnelling decay ∝ exp(−2κa), E={Etun}<V₀", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(_ARTIFACTS, "quantum_tunnelling.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("QUANTUM TUNNELLING validated — T(E) EMERGES from wave-matching (genuinely non-tautological):")
        print(f"  • the transfer matrix reproduces the analytic barrier transmission ({e1:.0e}); below the barrier a particle still")
        print(f"    gets through with T∝exp(−2κa) (slope {e2*100:.0f}% off the predicted −2κ) — the classically-forbidden tunnel;")
        print(f"  • above the barrier, the resonance peak ENERGIES (found by peak-detection on the T(E) curve) match")
        print(f"    V₀+(nπ/a)²/2 to {pos_err*100:.0f}% with real dips between — the wave nature, measured (positions emerge, not an identity).")
        print(f"  ⇒ the quantum SCATTERING tier: STM/tunnel junctions, alpha decay, resonant transport — extends the molecular-")
        print(f"    fidelity quantum tier (bound states in schrodinger_1d.py) with barrier penetration & resonance.")
    else:
        print(f"  (1)analytic {ok1} (2)tunnelling {ok2} (3)resonance {ok3} (4)limits {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
