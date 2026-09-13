"""COUPLED PIEZOELECTRIC ROD RESONATOR — the GENUINE version of piezoelectric.py gate (3). The audit found that gate fed
the precomputed open-circuit modulus c^D into an UNCOUPLED rod (the coupling never entered the eigenproblem). Here the
electrical DOFs are assembled INTO the coupled FE system, and the open-circuit stiffening + the resonance up-shift
√(1+k²) EMERGE from a charge-free Schur condensation of the coupled matrix — the coupling is genuinely in the eigenproblem,
the answer falls out of the solve.

Coupled 1D rod, linear elements, DOFs = displacement u + electric potential φ at each node. Element coupled stiffness from
the piezo constitutive [T;D]=[[c^E, e],[e, −κ]][S; −∂φ/∂x] (B=±1/le difference operator for both u and φ); mass on u only:
  K_uu=c^E·B, K_uφ=e·B, K_φφ=−κ·B  (B = 1D element stiffness).  Mechanical BC: fixed-free rod (u_0=0).
  SHORT circuit: electrodes grounded → all φ=0 → eigensolve(K_uu, M) → f_short (uses c^E).
  OPEN  circuit: no external charge → K_φu u + K_φφ φ = 0 → φ=−K_φφ⁻¹K_φu u → condense → (K_uu−K_uφK_φφ⁻¹K_φu) → f_open.

FALSIFICATION: (1) SHORT-circuit modes match the c^E rod modal f_n=(2n−1)/(4L)·√(c^E/ρ) (fixed-free longitudinal — anchor
NOT in the code as numbers, emerges); (2) ★the OPEN-circuit condensed stiffness EQUALS the c^D=c^E(1+k²) rod (the coupling
e,κ enters the eigenproblem and the Schur complement raises the modulus — measured, not plugged); (3) ★f_open/f_short=√(1+k²)
EMERGES from the coupled condensation; (4) NULL e=0 → f_open=f_short (decoupled). Render → artifacts/piezo_resonator.png.

  python em/piezo_resonator.py
"""
import os
import sys
import numpy as np

_ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
from scipy.linalg import eigh


def assemble(nel, L, cE, e, kap, rho):
    """coupled piezo rod: returns Kuu, Kuf, Kff (electrical), M (mass on u). nel elements, nn=nel+1 nodes."""
    le = L / nel; nn = nel + 1
    Be = (1.0 / le) * np.array([[1.0, -1.0], [-1.0, 1.0]])     # 1D element stiffness shape (BᵀB·le)
    Me = rho * le / 6 * np.array([[2.0, 1.0], [1.0, 2.0]])
    Kuu = np.zeros((nn, nn)); Kuf = np.zeros((nn, nn)); Kff = np.zeros((nn, nn)); M = np.zeros((nn, nn))
    for el in range(nel):
        d = [el, el + 1]
        for a in range(2):
            for b in range(2):
                Kuu[d[a], d[b]] += cE * Be[a, b]
                Kuf[d[a], d[b]] += e * Be[a, b]
                Kff[d[a], d[b]] += -kap * Be[a, b]
                M[d[a], d[b]] += Me[a, b]
    return Kuu, Kuf, Kff, M


def freqs(nel, L, cE, e, kap, rho, mode):
    """fundamental + first few longitudinal frequencies. mode='short' (phi=0) or 'open' (charge-free condensation)."""
    Kuu, Kuf, Kff, M = assemble(nel, L, cE, e, kap, rho)
    fix = [0]                                                  # mechanical fixed-free: u_0 = 0
    um = [i for i in range(nel + 1) if i not in fix]
    if mode == 'short':
        Keff = Kuu[np.ix_(um, um)]
    else:                                                      # open: ground phi_0 (gauge), condense the free phi DOFs
        pf = list(range(1, nel + 1))                           # free electrical DOFs (phi_0=0 reference)
        Kff_i = np.linalg.inv(Kff[np.ix_(pf, pf)])
        Kcond = Kuu - Kuf[:, pf] @ Kff_i @ Kuf[:, pf].T        # Schur complement (Kfu = Kuf^T since symmetric coupling)
        Keff = Kcond[np.ix_(um, um)]
    Mm = M[np.ix_(um, um)]
    w2 = np.sort(np.maximum(eigh(Keff, Mm, eigvals_only=True), 0.0))
    return np.sqrt(w2[:3]) / (2 * np.pi)


def main():
    print("=" * 84)
    print("COUPLED PIEZOELECTRIC ROD RESONATOR — open-circuit stiffening EMERGES from the coupled eigenproblem")
    print("=" * 84)
    L, cE, e, kap, rho, nel = 1.0, 100.0, 5.0, 1.0, 1.0, 120
    k2 = e ** 2 / (cE * kap); cD = cE * (1 + k2)
    print(f"\n  rod: c^E={cE} e={e} κ={kap} ρ={rho};  k²=e²/(c^E κ)={k2:.4f};  c^D=c^E(1+k²)={cD:.2f}")

    f_short = freqs(nel, L, cE, e, kap, rho, 'short')
    f_open = freqs(nel, L, cE, e, kap, rho, 'open')

    # (1) short-circuit fundamental matches the c^E fixed-free longitudinal rod f_1=(1/4L)√(c^E/ρ)
    f1_an = 1.0 / (4 * L) * np.sqrt(cE / rho)
    e1 = abs(f_short[0] - f1_an) / f1_an; ok1 = e1 < 0.01
    print(f"\n  (1) SHORT-circuit f_1={f_short[0]:.4f} vs c^E rod (1/4L)√(c^E/ρ)={f1_an:.4f}  (Δ{e1*100:.2f}%)  {'✓' if ok1 else 'FAIL'}")

    # (2) the OPEN-circuit condensed modulus (back out c_eff from f_open) equals c^D — coupling entered the eigenproblem
    cD_meas = cD_from = (f_open[0] * 4 * L) ** 2 * rho
    e2 = abs(cD_meas - cD) / cD; ok2 = e2 < 0.01
    print(f"  (2) ★OPEN-circuit effective modulus (from f_open) = {cD_meas:.2f} vs c^D={cD:.2f} (Δ{e2*100:.2f}%) — the coupling")
    print(f"      (e,κ) entered the eigenproblem; the Schur condensation raised c^E→c^D, NOT plugged  {'✓' if ok2 else 'FAIL'}")

    # (3) ★ratio f_open/f_short = √(1+k²) EMERGES (per mode)
    ratio = f_open / f_short; exp = np.sqrt(1 + k2)
    e3 = np.max(np.abs(ratio - exp) / exp); ok3 = e3 < 0.01
    print(f"  (3) ★f_open/f_short per mode = {np.round(ratio,4)} vs √(1+k²)={exp:.4f} (max Δ{e3*100:.2f}%) — up-shift from the coupled solve  {'✓' if ok3 else 'FAIL'}")

    # (4) NULL e=0 → decoupled
    f_open0 = freqs(nel, L, cE, 0.0, kap, rho, 'open')
    ok4 = np.max(np.abs(f_open0 - f_short) / f_short) < 1e-9
    print(f"  (4) NULL (e=0): f_open=f_short (max Δ{np.max(np.abs(f_open0-f_short)/f_short):.1e}) — no coupling, no shift  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        ks = np.linspace(0, 0.6, 30); rr = []
        for kk in ks:
            ee = np.sqrt(kk * cE * kap); rr.append(freqs(40, L, cE, ee, kap, rho, 'open')[0] / freqs(40, L, cE, ee, kap, rho, 'short')[0])
        fig, ax = plt.subplots(figsize=(5.4, 3.6), dpi=120)
        ax.plot(ks, rr, "bo-", ms=3, label="f_open/f_short (coupled eigensolve)"); ax.plot(ks, np.sqrt(1 + ks), "r--", lw=0.8, label="√(1+k²)")
        ax.set_xlabel("k²"); ax.set_ylabel("resonance up-shift"); ax.set_title("piezo resonance shift — emergent from the coupled solve", fontsize=9); ax.legend(fontsize=7)
        fig.tight_layout(); os.makedirs(_ART, exist_ok=True); fig.savefig(os.path.join(_ART, "piezo_resonator.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("COUPLED PIEZO RESONATOR validated — the genuine eigenproblem (audit-corrected piezo gate 3):")
        print(f"  • the electrical DOFs are assembled INTO the coupled FE; the open-circuit modulus c^D ({cD_meas:.1f}) and the")
        print(f"    resonance up-shift √(1+k²)={exp:.3f} EMERGE from a charge-free Schur condensation — the coupling (e,κ) is")
        print(f"    genuinely in the eigenproblem, not plugged. NULL-clean. ⇒ the proper resonator model for piezo sensors/actuators.")
    else:
        print(f"  (1)short {ok1} (2)c^D-emerges {ok2} (3)ratio {ok3} (4)NULL {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
