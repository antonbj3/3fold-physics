"""TWO-LEVEL SYSTEM / PARAMAGNET — RENDER->MATCH the thermodynamics of the simplest quantum system: a spin that can
point only UP or DOWN in a field B, with energies -/+ mu B. Boltzmann-occupying the two states gives the magnetization
    M = N mu tanh(mu B / kT),
which (i) for weak field is LINEAR in B and falls as 1/T -- the CURIE LAW chi = N mu^2/(kT) = C/T, a paramagnet WEAKENS when
heated; (ii) SATURATES at N mu when every spin is aligned. The heat capacity, from the energy variance, is NOT monotone: it RISES
from zero (frozen), PEAKS near kT ~ 0.83 mu B, and falls again -- the SCHOTTKY ANOMALY, the thermal fingerprint of a finite level
gap (no equipartition for a bounded spectrum). We evaluate the two-state partition function and recover the magnetization, the
Curie law, the saturation, and the Schottky peak. render_match.

MATCH: the magnetization emerges as N mu tanh(mu B/kT); no field gives no magnetization; strong field saturates at N mu; the susceptibility follows the Curie law chi=C/T and the heat capacity shows the Schottky peak.
  python quantum/two_level_paramagnet.py
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
import numpy as np
from render_match_scaffold import Benchmark, render_match

MU, N = 1.0, 1.0                                          # magnetic moment, spin count (k_B=1)


def thermo(B, T):
    """two-state (E=-/+ mu B) partition function -> (magnetization, heat capacity) by genuine Boltzmann sum."""
    E = np.array([-MU * B, MU * B]); w = np.exp(-E / T); Z = w.sum()
    M = N * (MU * w[0] - MU * w[1]) / Z
    C = N * ((E ** 2 * w).sum() / Z - ((E * w).sum() / Z) ** 2) / T ** 2   # var(E)/kT^2
    return float(M), float(C)


def magnetization(B, T=1.0):
    return thermo(B, T)[0]


def main():
    print("=" * 96)
    print("TWO-LEVEL PARAMAGNET — M=N mu tanh(mu B/kT), Curie law, Schottky anomaly; render->match")
    print("=" * 96)
    B0 = 1.0
    def rfn(p):
        return magnetization(p.get("B", B0), p.get("T", 1.0))
    band = [{"B": 0.5}, {"B": 2.0}]                      # field-strength uncertainty = sigma
    bench = N * MU * np.tanh(MU * B0 / 1.0)
    res = render_match(
        rfn, band, {"B": B0},
        Benchmark("magnetization M", round(bench, 4), 0.1, "N mu tanh(mu B/kT) two-level (EXTERNAL)", "mu"),
        nulls=[("with no field the spins point every which way -- zero net magnetization (B->0 -> M->0)", {"B": 1e-4}, lambda v, m: v < m / 50)],
        perturbations=[("a stronger field aligns more spins, saturating at N mu (B up -> larger M)", {"B": 5.0}, lambda v, best: v > best)],
        notes=["the magnetization is N mu tanh(mu B/kT); no field gives no magnetization, and a strong field saturates at N mu"])
    print(res.report())
    # ★the Curie law + saturation + Schottky anomaly
    m0 = magnetization(B0)
    chi = lambda T: magnetization(0.01, T) / 0.01
    Ts = np.linspace(0.2, 4, 60); Cs = np.array([thermo(1.0, T)[1] for T in Ts]); ipk = int(np.argmax(Cs))
    print(f"\n  B -> M:  " + "  ".join(f"{b:.1f}:{magnetization(b):.3f}" for b in (0.1, 1.0, 5.0)) + f"   (saturates at N mu={N*MU})")
    print(f"  M (B=1,T=1)={m0:.4f}  vs N mu tanh={bench:.4f}  (match {abs(m0-bench)/bench*100:.2f}%)")
    print(f"  ★CURIE LAW chi=M/B ~ C/T: T=0.5 -> {chi(0.5):.3f}, T=1 -> {chi(1.0):.3f}, T=2 -> {chi(2.0):.3f} (vs N mu^2/kT = {N*MU**2/0.5:.1f},{N*MU**2/1.0:.1f},{N*MU**2/2.0:.1f}) -- paramagnet WEAKENS when heated")
    print(f"  ★SCHOTTKY ANOMALY: heat capacity peaks C={Cs[ipk]:.4f} at kT={Ts[ipk]:.3f} (theory ~0.83 mu B=0.83); C->0 frozen ({thermo(1.0,0.2)[1]:.1e}) AND hot ({thermo(1.0,4.0)[1]:.3f})")
    print(f"  (4) ★ALL FROM TWO STATES: occupying just E=-/+ mu B by Boltzmann, the magnetization is N mu tanh(mu B/kT) to {abs(m0-bench)/bench*100:.0f}%; expanded for weak field it gives the Curie law chi=N mu^2/kT (susceptibility ~1/T, read off, not assumed), and the energy variance gives a heat capacity that PEAKS at kT~0.83 mu B -- the Schottky anomaly, unique to a system with a finite gap (an ideal gas has no such peak). A bounded spectrum cannot keep absorbing heat: C must turn over")
    print(f"  (5) ★WHY IT MATTERS -- THERMOMETRY, COOLING, THE SAME MATH AS RUBBER: the Schottky peak's position is a thermometer for level splittings (defects, nuclei, qubits), and adiabatic DEMAGNETIZATION rides the M(B/T) curve to reach milli-kelvin. The orientational partition function is identical to rubber_entropic's freely-jointed chain -- but the paramagnet's M~B/T WEAKENS with heat while the entropic spring's force~T STIFFENS: same Boltzmann machinery, opposite temperature dependence, because one fixes the field and the other the extension")
    g4 = abs(m0 - bench) / bench < 0.01 and abs(chi(2.0) - N * MU ** 2 / 2.0) / (N * MU ** 2 / 2.0) < 0.02 and abs(Ts[ipk] - 0.833) < 0.1  # tanh; Curie; Schottky peak
    g5 = magnetization(1e-4) < bench / 50 and magnetization(5.0) > m0 and magnetization(20.0) > 0.99 * N * MU  # null; rises; saturates at N mu
    ok = res.ok and g4 and g5
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (two-level paramagnet) — the magnetism of the simplest quantum system:")
        print(f"  • M={res.best:.4f}=N mu tanh(mu B/kT) (band [{res.band_lo:.3f},{res.band_hi:.3f}]=field σ), from the two-state partition function.")
        print(f"  • the susceptibility follows the Curie law chi=C/T (weakens when heated) and saturates at N mu; the heat capacity shows the Schottky peak at kT~0.83 mu B.")
        print(f"  • ★the same orientational Boltzmann math as the entropic spring -- but M~1/T (paramagnet) vs force~T (rubber): opposite temperature sign.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, tanh/Curie/Schottky {g4}, null/saturate {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
