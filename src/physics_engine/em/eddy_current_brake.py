"""EDDY-CURRENT BRAKE — A MAGNET FALLING THROUGH A CONDUCTING TUBE (electromagnetic-mechanical coupling) —
RENDER->MATCH the classic demo where a magnet dropped down a copper pipe drifts down in slow motion. As the magnet falls its field
sweeps past the tube wall; by Faraday's law the changing flux drives azimuthal EDDY CURRENTS, and by Lenz's law those currents
oppose the motion -- a velocity-proportional MAGNETIC DRAG that balances gravity at a terminal velocity. We do NOT assert the drag:
we COMPUTE the eddy-drag coefficient from the flux integral over the tube, INTEGRATE the falling magnet's equation of motion, and
MEASURE its terminal velocity. ★The numerical drag coefficient reproduces the exact Saslow result k = (45/1024) mu0^2 m^2 sigma w / a^4
(the dimensionless flux integral evaluates to 45/1024); ★the drag is LINEAR in velocity so the magnet approaches a TERMINAL velocity
exponentially (v_t = M g / k); ★the terminal velocity scales as a^4 in the tube radius and 1/m^2 in the dipole -- a wider pipe or a
weaker magnet falls much faster; ★with no magnetization (m=0) there are no eddy currents and the magnet is in FREE FALL (the null).
The tube radius a is the physical sigma. render_match_scaffold.

MATCH: the eddy-drag coefficient from the flux integral equals the Saslow law (45/1024) mu0^2 m^2 sigma w / a^4; ★the integrated falling-magnet motion reaches a terminal velocity (linear drag, cross-check) scaling as a^4 and 1/m^2; ★with m=0 there is no drag and the magnet free-falls (the null); a stronger magnet falls slower.
  python em/eddy_current_brake.py
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

MU0 = 4e-7 * np.pi
A_DEF = 0.012                                            # tube inner radius (the physical sigma) [m]
M_DIP = 0.3                                              # magnetic dipole moment [A m^2]
SIGMA_CU = 5.96e7                                        # copper conductivity [S/m]
W_WALL = 2e-3                                            # tube wall thickness [m]
MASS = 0.012                                             # magnet mass [kg]
G = 9.81
_cache = {}


def k_eddy(a=A_DEF, m=M_DIP):
    """eddy-drag coefficient from the flux integral k = (sigma w / 2 pi a) * integral( (dPhi/dz)^2 dz ) (memoized)."""
    key = (round(a, 9), round(m, 9))
    if key in _cache:
        return _cache[key]
    if m == 0.0:
        _cache[key] = 0.0; return 0.0
    z = np.linspace(-40 * a, 40 * a, 240000); dz = z[1] - z[0]
    dPhidz = MU0 * m * a ** 2 * (-3 * z) / (2 * (a ** 2 + z ** 2) ** 2.5)   # d/dz of on-axis-ring flux Phi=mu0 m a^2/(2(a^2+z^2)^1.5)
    k = SIGMA_CU * W_WALL / (2 * np.pi * a) * np.sum(dPhidz ** 2) * dz
    _cache[key] = k
    return k


def k_saslow(a=A_DEF, m=M_DIP):
    return (45.0 / 1024.0) * MU0 ** 2 * m ** 2 * SIGMA_CU * W_WALL / a ** 4   # exact (integral u^2/(1+u^2)^5 = 5pi/128)


def terminal_velocity(a=A_DEF, m=M_DIP):
    """integrate M dv/dt = M g - k v from rest; return the late-time terminal velocity (linear drag -> exponential approach)."""
    k = k_eddy(a, m)
    if k <= 0:
        return float(G * 5.0)                            # m=0: free fall -> v keeps growing; report v after 5 s (huge, no terminal)
    dt = 1e-4; v = 0.0; tau = MASS / k
    for _ in range(int(min(12 * tau, 50.0) / dt)):
        v += dt * (G - k * v / MASS)
    return v


def main():
    print("=" * 96)
    print("EDDY-CURRENT BRAKE — a magnet falling through a conducting tube; render->match")
    print("=" * 96)
    def rfn(p):
        return terminal_velocity(a=p.get("a", A_DEF), m=p.get("m", M_DIP))
    band = [{"a": 0.9 * A_DEF}, {"a": 1.1 * A_DEF}]      # tube-radius sigma: v_t scales as a^4 (two-sided)
    bench = MASS * G / k_saslow(A_DEF, M_DIP)           # Saslow terminal velocity (EXTERNAL analytic)
    res = render_match(
        rfn, band, {"a": A_DEF},
        Benchmark("terminal velocity", bench, 0.03 * bench, "Saslow M g a^4 1024/(45 mu0^2 m^2 sigma w) (EXTERNAL)", "m/s"),
        nulls=[("with NO magnetization (m=0) the falling object produces no changing flux, no eddy currents, no Lenz drag -- it is in FREE FALL and never reaches a terminal velocity (its speed just keeps growing as g t). The braking is ENTIRELY the induced eddy currents (m=0 -> no drag, free fall)", {"m": 0.0}, lambda v, mm: v > 5 * bench)],
        perturbations=[("a STRONGER magnet drives larger eddy currents and a larger Lenz drag (k ~ m^2), so it falls SLOWER -- the terminal velocity drops as 1/m^2 (double the dipole, quarter the terminal speed)", {"m": 2 * M_DIP}, lambda v, best: v < 0.4 * best)],
        notes=["the eddy-drag matches the Saslow 45/1024 law; the linear drag gives a terminal velocity scaling as a^4 and 1/m^2; m=0 is free fall"])
    print(res.report())
    kn, ks = k_eddy(), k_saslow()
    aas = [0.010, 0.012, 0.015]
    print(f"\n  ★DRAG COEFFICIENT FROM THE FLUX INTEGRAL (computed, not asserted): k={kn:.4e} vs Saslow (45/1024)mu0^2 m^2 sigma w/a^4={ks:.4e} ({abs(kn-ks)/ks*100:.2f}%) -- the dimensionless flux integral int (dPhi/dz)^2 evaluates to the exact 45/1024 (= 9/4 * 5pi/128 / (2pi)); no drag law entered the quadrature. C_dimensionless = k a^4/(mu0^2 m^2 sigma w) = {kn*A_DEF**4/(MU0**2*M_DIP**2*SIGMA_CU*W_WALL):.5f} vs 45/1024 = {45/1024:.5f}")
    print(f"  ★★IT REACHES A TERMINAL VELOCITY -- LINEAR DRAG (the falsifier): the equation of motion M dv/dt = M g - k v was integrated from rest; v rises and SATURATES at v_t = M g/k = {bench:.3f} m/s (an eddy brake gives velocity-proportional drag, hence a terminal velocity -- unlike viscous v^2 drag or no drag at all)")
    print(f"  ★a^4 SCALING IN THE TUBE RADIUS (the sigma): v_t vs a -- " + ", ".join(f"a={a*1000:.0f}mm:{terminal_velocity(a=a):.3f}(v/a^4={terminal_velocity(a=a)/a**4:.2e})" for a in aas) + ". The terminal velocity rises as a^4: a slightly wider pipe brakes far more weakly (the dipole field couples to the wall as 1/a^4), which is why the demo needs a snug-fitting magnet")
    print(f"  (4) ★WHY IT MATTERS: the eddy-current brake is contactless, wear-free velocity-proportional damping -- it stops roller-coasters and trains (linear induction brakes), damps tuned-mass dampers in skyscrapers and seismometers, brakes free-fall rides, and is the loss mechanism in maglev and in any conductor moving through a field. The Saslow 45/1024 a^4 law and the linear-drag terminal velocity are the magnetic-damping closure an electromechanical twin integrates")
    Cdim = kn * A_DEF ** 4 / (MU0 ** 2 * M_DIP ** 2 * SIGMA_CU * W_WALL)
    g4 = abs(kn - ks) / ks < 0.02 and abs(Cdim - 45 / 1024) / (45 / 1024) < 0.02 and terminal_velocity(m=0.0) > 5 * bench   # Saslow match; free-fall null
    g5 = terminal_velocity(m=2 * M_DIP) < 0.3 * bench and abs(terminal_velocity(a=0.015) / 0.015 ** 4 - terminal_velocity(a=0.010) / 0.010 ** 4) / (terminal_velocity(a=0.010) / 0.010 ** 4) < 0.02   # 1/m^2; a^4 scaling
    ok = res.ok and g4 and g5
    import os, json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/eddy_current_brake.json", "w") as fh:
        json.dump({"module": "eddy_current_brake", "provenance": "self-contained eddy-drag flux integral + falling-magnet ODE, no external data",
                   "k_numerical": kn, "k_saslow": ks, "C_dimensionless": Cdim, "saslow_constant": 45 / 1024,
                   "terminal_velocity": bench, "a": A_DEF, "dipole": M_DIP,
                   "vt_vs_a": {f"{a}": {"v_t": float(terminal_velocity(a=a)), "v_over_a4": float(terminal_velocity(a=a) / a ** 4)} for a in aas},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"saslow_null": bool(g4), "msq_a4": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (eddy-current brake) — Lenz drag brakes a falling magnet to a terminal velocity:")
        print(f"  • eddy-drag coefficient matches the Saslow (45/1024) mu0^2 m^2 sigma w/a^4 law to {abs(kn-ks)/ks*100:.2f}% (band=tube-radius σ).")
        print(f"  • ★linear drag -> terminal velocity v_t={bench:.3f} m/s scaling as a^4, 1/m^2; m=0 is free fall (the null).")
        print(f"  • ★a distinct electromagnetic-mechanical primitive for maglev/induction-brake/vibration-damping twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, saslow/null {g4}, m^2/a^4 {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
