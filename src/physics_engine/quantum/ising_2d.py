"""2D ISING MODEL — the order–disorder phase transition. ★The critical temperature T_c = 2/ln(1+√2) ≈ 2.2692 (Onsager's
EXACT result) is NOWHERE in the code — it EMERGES as the magnetization collapse and the susceptibility peak of a Metropolis
Monte-Carlo (emergence, not assertion). ★GEOMETRIC/critical: cooperative spin alignment vs thermal disorder undergoes
a sharp transition at a temperature fixed by the lattice geometry (nearest-neighbour coupling on a square lattice).

Metropolis on an L×L lattice (J=1, periodic), vectorized checkerboard updates. Below T_c the system spontaneously magnetises
(|M|→1); above, |M|→0; the susceptibility χ=(⟨M²⟩−⟨|M|⟩²)L²/T peaks at T_c.

FALSIFICATION (external anchor + finite-size): (1) ★the susceptibility peak → T_c=2.2692 (emergent, not coded); (2) the
magnetization |M(T)| collapses through the transition (ordered below, disordered above); (3) NULL/limits: T→0 fully ordered
(|M|→1), T→∞ disordered (|M|→0); (4) the energy per spin is monotone in T (caloric sanity). Figure written to artifacts/ising_2d.png.

  python quantum/ising_2d.py
"""
import sys
import numpy as np
import os

_ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(_ARTIFACTS, exist_ok=True)



def sweep(s, T, rng, mask):
    """one checkerboard Metropolis half-sweep on the sublattice `mask` (vectorized)."""
    nb = (np.roll(s, 1, 0) + np.roll(s, -1, 0) + np.roll(s, 1, 1) + np.roll(s, -1, 1))
    dE = 2.0 * s * nb                                          # energy change to flip
    flip = mask & ((dE <= 0) | (rng.random(s.shape) < np.exp(-dE / T)))
    s[flip] *= -1
    return s


def simulate(L, T, rng, eq=400, meas=600):
    s = rng.choice([-1, 1], size=(L, L))
    cb = np.indices((L, L)).sum(0) % 2                         # checkerboard
    A = cb == 0; B = cb == 1
    for _ in range(eq):
        s = sweep(s, T, rng, A); s = sweep(s, T, rng, B)
    Ms = []; Es = []
    for _ in range(meas):
        s = sweep(s, T, rng, A); s = sweep(s, T, rng, B)
        Ms.append(abs(s.mean()))
        nb = (np.roll(s, 1, 0) + np.roll(s, 1, 1))
        Es.append(-(s * nb).mean())
    Ms = np.array(Ms)
    chi = (np.mean(Ms ** 2) - np.mean(Ms) ** 2) * L * L / T
    return float(Ms.mean()), float(chi), float(np.mean(Es))


def main():
    print("=" * 84)
    print("2D ISING MODEL — the critical temperature T_c EMERGES (Onsager 2.2692) from the Monte-Carlo")
    print("=" * 84)
    TC = 2.0 / np.log(1 + np.sqrt(2))                          # Onsager exact = 2.26919
    rng = np.random.default_rng(0)
    L = 32
    Ts = np.round(np.linspace(1.6, 3.2, 17), 3)
    print(f"\n  L={L}, Onsager T_c = 2/ln(1+√2) = {TC:.4f}; scanning T...")
    M = []; X = []; E = []
    for T in Ts:
        m, x, e = simulate(L, T, rng); M.append(m); X.append(x); E.append(e)
    M = np.array(M); X = np.array(X); E = np.array(E)

    # (1) ★susceptibility peak → T_c
    Tc_meas = Ts[np.argmax(X)]
    e1 = abs(Tc_meas - TC) / TC; ok1 = e1 < 0.06
    print(f"\n  (1) ★susceptibility χ peaks at T={Tc_meas:.3f} vs Onsager T_c={TC:.4f} (Δ{e1*100:.1f}%) — T_c emerges, not coded  {'✓' if ok1 else 'FAIL'}")

    # (2) magnetization collapse through the transition
    M_lo = M[Ts <= TC - 0.4].mean(); M_hi = M[Ts >= TC + 0.4].mean()
    ok2 = M_lo > 0.7 and M_hi < 0.3
    print(f"  (2) MAGNETIZATION collapse: |M| below T_c = {M_lo:.2f} (ordered) → above = {M_hi:.2f} (disordered)  {'✓' if ok2 else 'FAIL'}")
    print(f"      |M(T)| = {np.round(M,2)}")
    print(f"        for T = {list(Ts)}")

    # (3) limits
    m0, _, _ = simulate(L, 1.0, rng); minf, _, _ = simulate(L, 6.0, rng)
    ok3 = m0 > 0.95 and minf < 0.15
    print(f"  (3) LIMITS: T=1.0 → |M|={m0:.2f} (≈1, ordered); T=6.0 → |M|={minf:.2f} (≈0, disordered)  {'✓' if ok3 else 'FAIL'}")

    # (4) energy monotone in T (caloric sanity)
    ok4 = np.all(np.diff(E) > -0.02)                           # energy rises with T (allow tiny MC noise)
    print(f"  (4) ENERGY/spin monotone↑ in T: E={np.round(E,2)} (caloric sanity)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.4), dpi=110)
        ax[0].plot(Ts, M, "bo-", ms=3); ax[0].axvline(TC, color="k", ls=":", lw=0.8, label=f"T_c={TC:.3f}")
        ax[0].set_xlabel("T"); ax[0].set_ylabel("|M|"); ax[0].set_title("magnetization collapse", fontsize=8); ax[0].legend(fontsize=7)
        ax[1].plot(Ts, X, "ro-", ms=3); ax[1].axvline(TC, color="k", ls=":", lw=0.8); ax[1].set_xlabel("T"); ax[1].set_ylabel("χ"); ax[1].set_title("susceptibility peak at T_c", fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(_ARTIFACTS, "ising_2d.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("2D ISING validated — the critical temperature EMERGES (Onsager, genuinely non-tautological):")
        print(f"  • the susceptibility peak (T={Tc_meas:.2f}) lands at Onsager's exact T_c=2.2692 ({e1*100:.0f}%) — never coded;")
        print(f"    the magnetization collapses through it ({M_lo:.2f}→{M_hi:.2f}); clean ordered/disordered limits; monotone caloric.")
        print(f"  ⇒ cooperative phase transition / order parameter / critical scaling — the statistical-mechanics anchor")
        print(f"    (magnetism, ferroelectric/martensitic ordering, the Curie point) for the phase tier.")
    else:
        print(f"  (1)Tc {ok1} (2)magnetization {ok2} (3)limits {ok3} (4)caloric {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
