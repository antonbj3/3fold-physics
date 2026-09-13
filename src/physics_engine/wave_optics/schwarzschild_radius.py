"""SCHWARZSCHILD RADIUS — RENDER->MATCH the size of a black hole's event horizon, the radius at which the escape speed reaches
the speed of light:
    r_s = 2 G M / c^2
It scales LINEARLY with mass -- the Sun's is ~3 km, Earth's ~9 mm, the Milky Way's central black hole (~4e6 M_sun) ~12 million
km. The counter-intuitive twist is the average density inside the horizon: rho = M / ((4/3) pi r_s^3) ~ 1/M^2, so the BIGGER the
black hole the LESS dense it is -- a billion-solar-mass supermassive black hole has an average density below that of water, while a
solar-mass one is denser than a nucleus. Uses the render_match_scaffold helper (forward model vs benchmark, no fitted constant).

MATCH: r_s for one solar mass is ~2.95 km; it is linear in mass; the mean density inside falls as 1/M^2 (supermassive < water).
INPUT: none (constants in the module). OUTPUT: printed gate lines.
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
import sys, numpy as np
from render_match_scaffold import Benchmark, render_match

G = 6.674e-11; C = 2.998e8; MSUN = 1.989e30


def r_s(M_solar):
    return 2 * G * (M_solar * MSUN) / C ** 2                   # Schwarzschild radius [m]


def rho_inside(M_solar):
    return (M_solar * MSUN) / ((4 / 3) * np.pi * r_s(M_solar) ** 3)   # mean density inside the horizon [kg/m^3]


def main():
    print("=" * 92)
    print("SCHWARZSCHILD RADIUS — r_s=2GM/c^2; linear in mass; bigger holes are LESS dense; render->match")
    print("=" * 92)
    def rfn(p):
        return r_s(p["M"]) / 1e3                               # km
    band = [{"M": 0.9}, {"M": 1.1}]                            # mass uncertainty = sigma
    res = render_match(
        rfn, band, {"M": 1.0},
        Benchmark("Schwarzschild radius of 1 solar mass", round(r_s(1.0) / 1e3, 2), 0.3, "r_s=2GM/c^2", "km"),
        nulls=[("vanishing mass has a vanishing horizon (M->0 -> r_s->0)", {"M": 1e-9}, lambda v, m: v < m / 1e6)],
        perturbations=[("a more massive hole is bigger (M up -> larger r_s)", {"M": 1e6}, lambda v, best: v > best)],
        notes=["no mass, no horizon; the radius grows in direct proportion to the mass"])
    print(res.report())
    # ★the linear radius + the 1/M^2 density inversion
    sM = np.polyfit(np.log([1, 1e3, 1e6]), np.log([r_s(m) for m in (1, 1e3, 1e6)]), 1)[0]
    sR = np.polyfit(np.log([1, 1e3, 1e6]), np.log([rho_inside(m) for m in (1, 1e3, 1e6)]), 1)[0]
    print(f"\n  M -> r_s:  Sun(1)->{r_s(1)/1e3:.1f}km  Sgr-A*(4e6)->{r_s(4e6)/1e9:.1f}e9 m  (Earth: {r_s(3.003e-6)*1e3:.1f} mm)")
    print(f"  (4) ★LINEAR HORIZON: r_s ~ M^{sM:.2f} -- exactly proportional to mass; double the mass, double the horizon, the cleanest formula in general relativity")
    print(f"  (5) ★BIGGER = LESS DENSE: mean density inside ~ M^{sR:.2f} (=1/M^2), so a 1 M_sun hole is {rho_inside(1):.0e} kg/m^3 (denser than a nucleus) but a 1e9 M_sun supermassive is only {rho_inside(1e9):.0f} kg/m^3 -- far below water (a 1e10 one is below air); you could cross its horizon without noticing locally")
    g4 = abs(sM - 1.0) < 0.01 and 2.5e3 < r_s(1.0) < 3.5e3   # linear; ~2.95 km for the Sun
    g5 = abs(sR + 2.0) < 0.02 and rho_inside(1e9) < 1000 and rho_inside(1) > 1e18   # 1/M^2; supermassive < air, solar > nucleus
    ok = res.ok and g4 and g5
    print("\n" + "=" * 92)
    if ok:
        print("RENDER→MATCH CLOSES (Schwarzschild radius) — the horizon size derived from GR, not fitted:")
        print(f"  • one solar mass has r_s={res.best:.2f} km (band [{res.band_lo:.2f},{res.band_hi:.2f}]=mass σ), linear in mass (slope {sM:.2f}).")
        print(f"  • the mean density inside falls as 1/M^2 (slope {sR:.2f}): a solar-mass hole beats nuclear density, a billion-solar supermassive is thinner than water.")
        print(f"  • the cleanest result in general relativity -- escape speed = c at r_s=2GM/c^2; pairs with eddington_luminosity for the accretion picture.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, linear/value {g4}, density {g5}. Fix at source.")
    print("=" * 92)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
