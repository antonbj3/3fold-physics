"""BLOCH OSCILLATION — RENDER->MATCH the counterintuitive fact that an electron in a crystal, pushed by a CONSTANT
force, does NOT accelerate away -- it OSCILLATES back and forth. A free electron under a constant force speeds up without bound
(x ~ t^2); but in a periodic lattice the crystal momentum k just rises linearly and WRAPS AROUND the Brillouin zone, so the band
velocity v(k)=dE/dk -- which reverses sign across the zone -- carries the electron forward then backward. The motion is BOUNDED,
periodic in time and space:
    Bloch period  T_B = h/(F a),     real-space amplitude  L_B = (band width)/F = 4J/(F a)   (cosine band -2J cos ka).
We do NOT type the closed-form band velocity: we build the tight-binding WANNIER-STARK Hamiltonian (hopping -J between sites +
a linear force potential F a n), diagonalise it, EVOLVE a real wavepacket psi(t)=exp(-iHt)psi0, and MEASURE the centre-of-mass
oscillation <x(t)> = sum n |psi_n(t)|^2 -- the Bloch oscillation EMERGES from the quantum dynamics. It CONVERGES (as the packet
narrows in k, i.e. widens in real space) to the semiclassical L_B = 4J/(F a) with a shrinking finite-width gap. Compared against a
FREE particle (parabolic band, no zone) which runs away. Gates run through render_match_scaffold.

MATCH: the tight-binding wavepacket's centre-of-mass Bloch amplitude converges to (band width)/F = 4J/(F a); a free particle (no band, no Brillouin zone) instead runs away unbounded; a stronger force gives a smaller amplitude.
  python quantum/bloch_oscillation.py
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

J, A, F0 = 1.0, 1.0, 1.0                                  # hopping (band half-width 2J); lattice constant; constant force


def bloch_amplitude(F=F0, Nsite=120, w=12.0, nt=240):
    """GENUINE tight-binding Wannier-Stark: diagonalise H = -J(hop) + F a n, evolve a wavepacket, MEASURE the centre-of-mass
    Bloch oscillation amplitude max<x> - min<x>. The bounded oscillation EMERGES -- no band velocity is typed."""
    n = np.arange(-Nsite, Nsite + 1).astype(float)
    Ns = len(n)
    H = np.diag(F * A * n) + np.diag(-J * np.ones(Ns - 1), 1) + np.diag(-J * np.ones(Ns - 1), -1)
    E, V = np.linalg.eigh(H)                              # Wannier-Stark ladder
    psi0 = np.exp(-n ** 2 / (2 * w ** 2)); psi0 /= np.linalg.norm(psi0)   # broad real wavepacket (narrow in k, k0=0)
    c = V.conj().T @ psi0
    T_B = 2 * np.pi / (F * A)
    ts = np.linspace(0.0, T_B, nt)
    xt = np.array([np.sum(n * np.abs(V @ (np.exp(-1j * E * t) * c)) ** 2) for t in ts])
    return float(xt.max() - xt.min())


def free_amplitude(F=F0, periods=4):
    """a FREE particle (parabolic band, no Brillouin zone to wrap): the crystal momentum keeps rising, v ~ k(t) ~ F t, so the
    displacement runs away x ~ t^2 -- unbounded, no oscillation. The contrast that makes the lattice's bounded motion remarkable."""
    T_B = 2 * np.pi / (F * A)
    t = np.linspace(0.0, periods * T_B, 4000)
    v = 2 * J * A ** 2 * (F * A * t)                     # parabolic band E=J(ka)^2: v=dE/dk ~ k(t)=F a t (no zone reversal)
    x = np.cumsum(v) * (t[1] - t[0])
    return float(x.max() - x.min())


def main():
    print("=" * 96)
    print("BLOCH OSCILLATION — constant force -> bounded oscillation (tight-binding wavepacket); amplitude = bandwidth/F")
    print("=" * 96)
    def rfn(p):
        return free_amplitude(F=p.get("F", F0)) if p.get("free") else bloch_amplitude(F=p.get("F", F0))
    band = [{"F": 1.1 * F0}, {"F": 0.9 * F0}]            # force σ (±10%): amplitude L_B = 4J/(F a) ~ 1/F
    res = render_match(
        rfn, band, {"F": F0},
        Benchmark("Bloch oscillation amplitude L_B", 4 * J / (F0 * A), 0.05, "(band width)/F = 4J/(F a) semiclassical limit (analytic)", "a"),
        nulls=[("a FREE particle (parabolic band, no Brillouin zone to wrap) just ACCELERATES -- the displacement runs away unbounded (~t^2), there is no oscillation amplitude to speak of (free -> amplitude diverges, >> the bounded L_B)", {"free": True}, lambda v, m: v > 5 * m)],
        perturbations=[("a STRONGER force sweeps k through the zone faster, so the electron turns around sooner -- a SMALLER real-space amplitude (F up -> L_B down)", {"F": 2 * F0}, lambda v, best: v < best - 1.0)],
        notes=["the tight-binding wavepacket Bloch amplitude converges to (band width)/F = 4J/(F a); a free particle runs away unbounded; a stronger force shrinks the amplitude"])
    print(res.report())
    # ★the bounded paradox + the finite-width CONVERGENCE to the semiclassical limit (the genuine-limit signature)
    L = bloch_amplitude()
    conv = {w: bloch_amplitude(w=w) for w in (4.0, 8.0, 16.0, 32.0)}
    gaps = {w: abs(v - 4 * J / (F0 * A)) / (4 * J / (F0 * A)) for w, v in conv.items()}
    g_conv = gaps[4.0] > gaps[16.0] > gaps[32.0] and gaps[32.0] < 5e-3
    print(f"\n  ★BLOCH AMPLITUDE = TIGHT-BINDING WAVEPACKET COM (evolved, not asserted): diagonalising the Wannier-Stark H and evolving a wavepacket, the centre of mass <x(t)> oscillates with amplitude {L:.4f} a vs the semiclassical (band width)/F = {4*J/(F0*A):.4f} ({abs(L-4*J/(F0*A))/(4*J/(F0*A))*100:.2f}%). The gap SHRINKS as the packet narrows in k (widens in real space) -- " + ", ".join(f"w={w:g}:{gaps[w]*100:.2f}%" for w in (4.0, 8.0, 16.0, 32.0)) + " -- a genuine finite-wavepacket-width correction converging to the semiclassical limit, NOT the closed-form velocity re-integrated")
    print(f"  ★A CONSTANT FORCE GIVES OSCILLATION, NOT ACCELERATION (the paradox): the electron displacement stays BOUNDED at {L:.2f} a, while a free particle under the same force reaches {free_amplitude():.0f} a and keeps going. The lattice converts steady push into back-and-forth motion -- the deepest signature of a band")
    print(f"  ★WHY IT TURNS AROUND (the Brillouin zone): k(t)=F t rises linearly and WRAPS the zone every T_B=h/(F a)={2*np.pi/(F0*A):.3f}; the band velocity reverses sign at the zone edge, so the electron decelerates, stops, and returns. No scattering, no wall -- pure band geometry, here read from the Wannier-Stark evolution")
    print(f"  ★AMPLITUDE ~ 1/F (band width / force): L_B = " + ", ".join(f"F={f}:{bloch_amplitude(F=f):.2f}" for f in (0.5, 1.0, 2.0)) + " = 4J/(F a); a weaker force lets the electron travel further before the zone turns it around. The amplitude measures the band width directly")
    print(f"  (4) ★WHY IT MATTERS: Bloch oscillations explain why a perfect crystal in a DC field carries no steady current (it would oscillate, not conduct) -- real conduction needs scattering to interrupt them; they were finally seen directly in semiconductor superlattices and cold atoms in optical lattices (a clean realization), and underlie Bloch-oscillation-based force/gravity sensors and the Wannier-Stark ladder")
    L2 = bloch_amplitude(F=2 * F0)
    g4 = res.ok and abs(L - 4 * J / (F0 * A)) / (4 * J / (F0 * A)) < 0.02 and g_conv and bloch_amplitude(F=2 * F0) < L < bloch_amplitude(F=0.5 * F0)  # amplitude=4J/F; converges; 1/F monotone
    g5 = free_amplitude() > 5 * L and abs(L2 - 2 * J / (F0 * A)) / (2 * J / (F0 * A)) < 0.02  # free unbounded; F=2 gives 4J/2F=2J/F
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/bloch_oscillation.json", "w") as fh:
        json.dump({"module": "bloch_oscillation", "provenance": "self-contained: GENUINE tight-binding Wannier-Stark diagonalisation + wavepacket evolution -> centre-of-mass Bloch oscillation amplitude EMERGES; converges to the semiclassical 4J/(F a) with a finite-wavepacket-width correction (no band velocity typed)",
                   "J": J, "a": A, "F": F0, "bloch_amplitude": L, "analytic_4J_over_Fa": 4 * J / (F0 * A), "gap_frac": abs(L - 4 * J / (F0 * A)) / (4 * J / (F0 * A)),
                   "convergence_vs_packet_width": {str(w): {"amp": conv[w], "gap": gaps[w]} for w in conv},
                   "free_amplitude": free_amplitude(), "amplitude_2F": L2,
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"amplitude_convergence_monotone": bool(g4), "free_null_scaling": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Bloch oscillation) — a constant force that makes an electron oscillate:")
        print(f"  • tight-binding wavepacket COM amplitude {res.best:.4f} a -> (band width)/F = 4J/(F a) ({abs(L-4*J/(F0*A))/(4*J/(F0*A))*100:.2f}%, band=force σ); finite-width gap shrinks {gaps[4.0]*100:.1f}→{gaps[32.0]*100:.2f}%.")
        print(f"  • bounded periodic motion (T_B=h/(F a)); a free particle runs away; amplitude ~ 1/F measures the band width.")
        print(f"  • ★the lattice turns a steady push into back-and-forth oscillation -- why a perfect crystal carries no DC current.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, amplitude/converge/monotone {g4}, free-null/scaling {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
