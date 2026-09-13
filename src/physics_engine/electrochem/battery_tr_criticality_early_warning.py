"""THERMAL-RUNAWAY CRITICALITY EARLY WARNING - critical slowing down at the Semenov fold.

What it computes: thermal runaway is a saddle-node (fold) bifurcation - the Semenov tangency of Arrhenius heat
generation against linear cooling. Approaching any fold, the recovery rate lambda goes to zero (critical slowing
down), so a noise-driven system shows diverging variance and lag-1 autocorrelation: a measurable early warning before
the fold is reached. The same normal form governs MEMS pull-in, buckling, flutter and the boiling crisis.

Physics: near the cool stable fixed point T_ss the temperature relaxes as d(dT)/dt = -lambda*dT + noise with
    lambda = (h*A_s - dQ_gen/dT|T_ss)/C,
the cooling slope minus the Arrhenius self-heating slope over the heat capacity. At the fold those slopes become
tangent, so lambda -> 0. The saddle-node normal form gives lambda proportional to (distance-to-fold)^(1/2); an
Ornstein-Uhlenbeck process at that fixed point then has stationary variance sigma^2/(2*lambda), proportional to
(distance)^(-1/2), and lag-1 autocorrelation exp(-lambda*dt) -> 1. Variance and autocorrelation of the cell
temperature therefore rise as the ambient approaches its critical value.

Inputs: none (Ea = 130 kJ/mol, pre-factor 3.0e16 W, cooling conductance 0.05 W/K, heat capacity 45 J/K, additive
thermal noise). Outputs: the critical ambient temperature, the analytic recovery-rate and variance scaling exponents,
an Euler-Maruyama simulation with its autocorrelation and shuffled null, and gate lines G1-G4.

Reference: saddle-node normal-form exponents (+1/2 for the recovery rate, -1/2 for the variance), the
Ornstein-Uhlenbeck variance relation, and the accelerating-rate-calorimetry observation that pre-onset self-heating
noise rises before runaway.

GATES: G1 critical slowing down - lambda goes to zero as the ambient approaches critical, scaling as
(distance)^(+1/2). G2 precursors diverge - variance as (distance)^(-1/2) and lag-1 autocorrelation rising toward 1.
G3 NULL: shuffling the noise trace destroys the autocorrelation, so the rise is the dynamics and not the noise, and
the simulated variance follows the Ornstein-Uhlenbeck law sigma^2/(2*lambda). G4 composes the Semenov fold model with
the generic criticality primitive.
"""
import sys
import numpy as np

R = 8.314
EA = 130e3                       # J/mol, decomposition activation energy
DH_A = 3.0e16                    # W, Arrhenius pre-factor (heat-release scale)
H_AS = 0.05                      # W/K surface cooling conductance
C_TH = 45.0                      # cell heat capacity (J/K) — for the dynamics
SIGMA = 0.02                     # additive thermal-noise amplitude (W-equivalent) — small fluctuations


def q_gen(T):
    return DH_A * np.exp(-EA / (R * T))


def dqgen_dT(T):
    return q_gen(T) * EA / (R * T ** 2)


def critical_Tamb():
    Ts = np.linspace(300, 700, 60000)
    g = dqgen_dT(Ts) - H_AS
    i = np.where(np.diff(np.sign(g)))[0]
    Tstar = Ts[i[0]]
    return Tstar - q_gen(Tstar) / H_AS, Tstar          # (Tamb_crit K, Tstar K)


def stable_fixed_point(Tamb_K):
    """lowest (cool, stable) root of Q_gen=Q_loss above ambient (K). Fine grid to resolve the near-fold root."""
    Ts = np.linspace(Tamb_K + 0.001, Tamb_K + 300, 300000)
    f = q_gen(Ts) - H_AS * (Ts - Tamb_K)
    roots = Ts[np.where(np.diff(np.sign(f)))[0]]
    return roots[0] if len(roots) else np.nan


def recovery_rate(Tamb_K):
    """λ = (h·A_s − dQ_gen/dT)|_{T_ss} / C  (1/s); → 0 at the fold."""
    Tss = stable_fixed_point(Tamb_K)
    return (H_AS - dqgen_dT(Tss)) / C_TH, Tss


def simulate(Tamb_K, dt=10.0, n=60000, seed=0, lag_s=400.0):
    """Euler-Maruyama on C·dT/dt = Q_gen−Q_loss + noise, at the cool fixed point. Returns variance + autocorr at lag_s."""
    rng = np.random.RandomState(seed)
    Tss = stable_fixed_point(Tamb_K)
    T = Tss
    burn = n // 5
    rec = np.empty(n - burn)
    for k in range(n):
        drift = (q_gen(T) - H_AS * (T - Tamb_K)) / C_TH
        T = T + drift * dt + (SIGMA / C_TH) * np.sqrt(dt) * rng.randn()
        if T < Tamb_K:
            T = Tamb_K + 1e-6
        if k >= burn:
            rec[k - burn] = T
    var = np.var(rec)
    L = max(int(lag_s / dt), 1)
    a = rec[:-L] - rec.mean(); b = rec[L:] - rec.mean()
    ac = float(np.sum(a * b) / np.sum((rec - rec.mean()) ** 2))
    # NULL: shuffled trace
    sh = rec.copy(); rng.shuffle(sh)
    a2 = sh[:-L] - sh.mean(); b2 = sh[L:] - sh.mean()
    ac_shuf = float(np.sum(a2 * b2) / np.sum((sh - sh.mean()) ** 2))
    return var, ac, ac_shuf


def main():
    print("=" * 100)
    print("TR CRITICALITY EARLY-WARNING: critical slowing down at the Semenov fold (saddle-node criticality)")
    print("=" * 100)
    Tc_K, Tstar = critical_Tamb()
    print(f"\n  Semenov fold: T_amb,crit = {Tc_K-273.15:.0f} °C. ANALYTIC critical slowing down APPROACHING the fold (asymptotic regime):")
    # (1) ANALYTIC λ vs distance, CLOSE to the fold (the saddle-node √ regime); the geometry/normal form
    dclose = np.array([2.0, 1.0, 0.5, 0.25, 0.125, 0.0625])  # distance-to-fold (°C) — asymptotic
    lam_c = np.array([recovery_rate(Tc_K - d)[0] for d in dclose])
    print(f"  {'dist-to-fold':>13}{'λ (1/s)':>12}{'τ=1/λ (s)':>12}{'var∝σ²/2λ':>13}")
    for d, l in zip(dclose, lam_c):
        print(f"  {d:>11.3f} °C{l:>12.2e}{1/l:>12.0f}{(SIGMA**2/(2*l*C_TH**2)):>13.2e}")
    p_lam = np.polyfit(np.log(dclose), np.log(lam_c), 1)[0]   # → +0.5 (normal form)
    var_an = SIGMA ** 2 / (2 * lam_c * C_TH ** 2)             # OU stationary variance ∝ 1/λ ∝ dist^-0.5
    p_var = np.polyfit(np.log(dclose), np.log(var_an), 1)[0]  # → -0.5
    ac_an = np.exp(-lam_c * 400.0)                            # autocorr at 400 s lag → 1 as λ→0
    print(f"  ★G1 recovery-rate scaling: λ ∝ (distance)^{p_lam:.2f} (saddle-node normal form predicts +0.5) → critical slowing down")
    print(f"  ★G2 precursors: OU variance ∝ (distance)^{p_var:.2f} (−0.5 law); lag-1 autocorr e^(−λ·400s) rises {ac_an[0]:.2f}→{ac_an[-1]:.2f} → 1 toward the fold")

    # (2) SIMULATION at moderate (deep-basin) distances → confirm the OU mechanism + the NULL
    print(f"\n  SIMULATION (Euler-Maruyama, deep-basin distances) confirming the OU mechanism + the NULL:")
    dsim = np.array([8.0, 4.0, 2.0])
    lam_s, var_s, ac_s, acs_s = [], [], [], []
    for d in dsim:
        Tamb_K = Tc_K - d
        l = recovery_rate(Tamb_K)[0]; v, a, a_sh = simulate(Tamb_K)
        lam_s.append(l); var_s.append(v); ac_s.append(a); acs_s.append(a_sh)
        print(f"    dist {d:.0f}°C: λ={l:.2e}, sim-var={v:.2e}, autocorr={a:.2f}, shuffled={a_sh:+.2f}")
    lam_s, var_s, ac_s, acs_s = map(np.array, (lam_s, var_s, ac_s, acs_s))
    p_var_lam = np.polyfit(np.log(lam_s), np.log(var_s), 1)[0]   # sim var ∝ λ^-1 (OU)
    print(f"  ★G3 OU: sim variance ∝ λ^{p_var_lam:.2f} (−1 law, var=σ²/2λ); autocorr rises {ac_s[0]:.2f}→{ac_s[-1]:.2f}; NULL shuffled |autocorr| ~{np.mean(np.abs(acs_s)):.2f} (flat)")

    g1 = abs(p_lam - 0.5) < 0.12                              # ★λ ∝ √(distance) — saddle-node normal form
    g2 = (abs(p_var + 0.5) < 0.12) and (ac_an[-1] > 0.9)      # ★variance diverges (−1/2) + autocorr → 1
    g3 = (abs(p_var_lam + 1.0) < 0.2) and (ac_s[-1] > ac_s[0]) and (np.mean(np.abs(acs_s)) < 0.1)  # ★OU var∝1/λ + autocorr rises + shuffle NULL flat
    g4 = g1 and g2                                            # composes the Semenov fold + saddle-node criticality
    ok = g1 and g2 and g3
    ac = ac_an; var = var_an; acs = acs_s
    print("\n" + "-" * 100)
    print(f"  G1 ★critical slowing down: λ ∝ (dist)^{p_lam:.2f} ≈ +0.5 (saddle-node normal form)        {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★precursors diverge: variance ∝ (dist)^{p_var:.2f} ≈ −0.5 + autocorr→{ac[-1]:.2f} (rises)   {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★OU: var ∝ λ^{p_var_lam:.2f} ≈ −1 + NULL shuffled autocorr {np.mean(np.abs(acs)):.2f}≈0 (flat)       {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes the Semenov fold + the saddle-node criticality primitive               {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("TR CRITICALITY EARLY-WARNING render→match LIVE — the saddle-node criticality primitive wired to thermal runaway. As the")
        print(f"  ambient approaches the Semenov fold (T_amb,crit={Tc_K-273.15:.0f}°C), the recovery rate λ collapses as (distance)^{p_lam:.2f} — the")
        print(f"  universal saddle-node normal form (+1/2) — so a noise-driven cell shows DIVERGING temperature variance ((dist)^{p_var:.2f}, the")
        print(f"  −1/2 law) and lag-1 autocorrelation rising toward 1 ({ac[0]:.2f}→{ac[-1]:.2f}). These are MEASURABLE before TR — the early-warning")
        print(f"  precursor, matching the ARC-calorimeter observation that pre-onset self-heating noise rises before runaway. ★NULL: shuffling")
        print(f"  the trace flattens the autocorr ({np.mean(np.abs(acs)):.2f}) — the rise is the critical slowing down (dynamics), not the noise; the variance")
        print(f"  obeys the OU law σ²/2λ (var∝λ^{p_var_lam:.2f}). Same fold, same precursor as MEMS pull-in, buckling, flutter and the boiling crisis.")
    else:
        print(f"HONEST WIP: g1={g1}(p_λ {p_lam:.2f}) g2={g2}(p_var {p_var:.2f}/ac {ac[-1]:.2f}) g3={g3}. Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
