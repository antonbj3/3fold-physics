"""1D TIME-INDEPENDENT SCHRÖDINGER — the energy spectrum EMERGES from diagonalizing a finite-difference Hamiltonian. ★The
quantized levels are NOWHERE in the code: the tridiagonal matrix H = −½ d²/dx² + V(x) contains only the potential and the
kinetic stencil; the eigenvalues E_n come out of the SOLVE (emergence, not assertion). This anchors the
Quantum tier — the basis for molecular-fidelity work (vibrational/electronic structure).

Two canonical potentials, each with an exact spectrum the code does not encode:
  • harmonic oscillator V=½ω²x²  → E_n = (n+½)ω  (EQUALLY SPACED levels — the phonon ladder);
  • infinite square well V=0 in [0,L] → E_n = n²π²/(2L²)  (the n² scaling — a confined mode).
ℏ=m=1 throughout. H built by finite differences, diagonalized with scipy.linalg.eigh_tridiagonal.

FALSIFICATION (external anchors not in the code): (1) ★HO levels E_n=(n+½)ω emerge (equal spacing = ω); (2) ★square-well
levels scale as n² and match n²π²/(2L²); (3) the HO ground state is the Gaussian exp(−½ωx²) (the wavefunction SHAPE emerges);
(4) NULL/sanity: eigenstates are orthonormal. Figure written to artifacts/schrodinger_1d.png.

  python quantum/schrodinger_1d.py
"""
import sys
import numpy as np
from scipy.integrate import trapezoid
from scipy.linalg import eigh_tridiagonal
import os

_ARTIFACTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
os.makedirs(_ARTIFACTS, exist_ok=True)



def solve(V, dx, k=6):
    """lowest k eigenpairs of H = -1/2 d²/dx² + V on a uniform grid (Dirichlet box). Returns E[k], psi[N,k]."""
    n = len(V)
    diag = 1.0 / dx ** 2 + V                     # -1/2 * (-2/dx²) = 1/dx² on the diagonal
    off = -0.5 / dx ** 2 * np.ones(n - 1)        # -1/2 * (1/dx²) off-diagonal
    E, psi = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
    psi = psi / np.sqrt(dx)                       # normalize ∫|ψ|²dx = 1
    return E, psi


def main():
    print("=" * 84)
    print("1D SCHRÖDINGER — the quantized energy spectrum EMERGES from diagonalizing H = −½∂²+V (quantum tier)")
    print("=" * 84)

    # --- harmonic oscillator: E_n = (n+½)ω ---
    w = 1.0
    L = 12.0; N = 2000
    x = np.linspace(-L / 2, L / 2, N); dx = x[1] - x[0]
    Eho, psi = solve(0.5 * w ** 2 * x ** 2, dx, k=6)
    En_an = (np.arange(6) + 0.5) * w
    e1 = np.max(np.abs(Eho - En_an)); ok1 = e1 < 5e-3
    print(f"\n  HARMONIC OSCILLATOR (ω={w}):")
    print(f"    emergent E_n = {np.round(Eho,4)}")
    print(f"    exact (n+½)ω = {np.round(En_an,4)}")
    spac = np.diff(Eho)
    e_sp = np.max(np.abs(spac - w))
    print(f"  (1) ★HO levels E_n=(n+½)ω emerge (max Δ{e1:.1e}); EQUAL SPACING ΔE={np.round(spac,4)}≈ω (Δ{e_sp:.1e}) — phonon ladder  {'✓' if ok1 else 'FAIL'}")

    # --- infinite square well: E_n = n²π²/(2L²) ---
    Lw = 1.0; Nw = 2000
    xw = np.linspace(0, Lw, Nw); dxw = xw[1] - xw[0]
    Esw, psiw = solve(np.zeros(Nw), dxw, k=5)
    n_idx = np.arange(1, 6)
    Esw_an = n_idx ** 2 * np.pi ** 2 / (2 * Lw ** 2)
    e2 = np.max(np.abs(Esw - Esw_an) / Esw_an); ok2 = e2 < 0.01
    ratios = Esw / Esw[0]                         # should be 1,4,9,16,25
    print(f"\n  INFINITE SQUARE WELL (L={Lw}):")
    print(f"    emergent E_n = {np.round(Esw,3)}  vs  n²π²/2L² = {np.round(Esw_an,3)}")
    print(f"  (2) ★square-well levels scale as n²: E_n/E_1 = {np.round(ratios,2)} (≈1,4,9,16,25); match {e2*100:.1f}%  {'✓' if ok2 else 'FAIL'}")

    # --- (3) HO ground state is the Gaussian exp(-½ωx²) ---
    g = np.exp(-0.5 * w * x ** 2); g = g / np.sqrt(trapezoid(g ** 2, x))
    psi0 = psi[:, 0] * np.sign(psi[:, 0][N // 2])   # fix sign
    e3 = np.max(np.abs(np.abs(psi0) - np.abs(g))) / np.abs(g).max(); ok3 = e3 < 0.01
    print(f"  (3) HO ground state = Gaussian exp(−½ωx²): max shape err {e3*100:.2f}% — the wavefunction shape emerges  {'✓' if ok3 else 'FAIL'}")

    # --- (4) orthonormality ---
    G = psi.T @ psi * dx
    ortho = np.max(np.abs(G - np.eye(6)))
    ok4 = ortho < 1e-6
    print(f"  (4) ORTHONORMALITY ⟨ψ_m|ψ_n⟩=δ_mn: max off-identity {ortho:.1e} (sanity)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.6), dpi=110)
        ax[0].plot(x, 0.5 * w ** 2 * x ** 2, "k-", lw=0.6)
        for n in range(5):
            p = psi[:, n] * np.sign(psi[:, n][N // 2 + (n % 2)])
            ax[0].plot(x, Eho[n] + 0.6 * p, lw=0.9); ax[0].axhline(Eho[n], color="grey", ls=":", lw=0.4)
        ax[0].set_xlim(-5, 5); ax[0].set_ylim(0, 6); ax[0].set_title("HO: equally-spaced E_n=(n+½)ω", fontsize=8); ax[0].set_xlabel("x")
        ax[1].plot(n_idx, Esw, "bo", label="emergent"); ax[1].plot(n_idx, Esw_an, "r--", lw=0.8, label="n²π²/2L²")
        ax[1].set_title("square well: E_n ∝ n²", fontsize=8); ax[1].set_xlabel("n"); ax[1].set_ylabel("E"); ax[1].legend(fontsize=7)
        fig.tight_layout(); fig.savefig(os.path.join(_ARTIFACTS, "schrodinger_1d.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("1D SCHRÖDINGER validated — the spectrum EMERGES from the Hamiltonian (genuinely non-tautological):")
        print(f"  • HO levels (n+½)ω come out EQUALLY SPACED (the phonon/vibrational ladder), the ground state is the Gaussian;")
        print(f"  • the square well gives the n² confined ladder — both spectra emerge from diagonalizing H, never coded.")
        print(f"  ⇒ the QUANTUM tier: vibrational modes, electronic structure, tunnelling — the foundation of the")
        print(f"    molecular-fidelity target (the deepest fidelity axis below continuum physics).")
    else:
        print(f"  (1)HO {ok1} (2)well {ok2} (3)groundstate {ok3} (4)orthonormal {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
