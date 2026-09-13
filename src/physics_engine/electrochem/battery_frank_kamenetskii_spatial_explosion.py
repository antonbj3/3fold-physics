"""FRANK-KAMENETSKII SPATIAL THERMAL EXPLOSION - the spatially resolved companion to the lumped Semenov criticality.

What it computes: a lumped (0-D) Semenov model assumes a uniform cell temperature. A large cell is not isothermal:
heat is generated everywhere but lost only at the surface, so the centre runs hottest and the criticality is spatial.
Frank-Kamenetskii theory (1939) gives the steady dimensionless temperature equation
    laplacian(theta) + delta*exp(theta) = 0,  theta(surface) = 0, theta'(centre) = 0,
in plane, cylindrical and spherical symmetry (j = 0, 1, 2). Rescaling r -> xi/sqrt(delta) removes delta; shooting from
the centre (theta(0) = theta0, theta'(0) = 0) to where theta = 0 gives xi0 and hence delta(theta0) = xi0^2. That curve
rises, turns over at a maximum and falls; the maximum is the critical delta, a saddle-node bifurcation: below it two
profiles exist (stable and unstable), above it none, which is ignition.

Inputs: none (boundary-value problem solved numerically). Outputs: the numerical critical delta per geometry with its
error against the textbook value, the geometry ordering, the turning-point check, and gate lines G1-G4.

Reference: Frank-Kamenetskii critical values delta_crit = 0.878 (infinite slab), 2.000 (infinite cylinder), 3.322
(sphere).

GATES: G1 the slab critical delta matches 0.878 to a few percent. G2 geometry ordering slab < cylinder < sphere, all
three matching the textbook values (the sphere is most stable, best surface-to-volume). G3 the critical delta is a
saddle-node turning point; above it no steady profile exists (the explosion); sigma is the numeric-versus-analytic
error. G4 extends the lumped Semenov criticality to its spatial form: a pouch (slab) ignites at far lower reaction
strength than a cylindrical cell.
"""
import sys
import numpy as np
from scipy.integrate import solve_ivp

FK_LIT = {0: ("slab", 0.878), 1: ("cylinder", 2.000), 2: ("sphere", 3.322)}


def delta_of_theta0(theta0, j):
    """shoot the FK ODE from the centre; return δ = ξ₀² where θ first hits 0."""
    xi0 = 1e-5
    th_s = theta0 - np.exp(theta0) * xi0 ** 2 / (2 * (j + 1))      # series start (θ'(0)=0)
    thp_s = -np.exp(theta0) * xi0 / (j + 1)

    def rhs(xi, y):
        th, thp = y
        return [thp, -np.exp(th) - (j / xi) * thp]

    def hit_zero(xi, y):
        return y[0]
    hit_zero.terminal = True; hit_zero.direction = -1
    sol = solve_ivp(rhs, [xi0, 60.0], [th_s, thp_s], events=hit_zero, rtol=1e-9, atol=1e-11, max_step=0.02)
    if sol.t_events[0].size:
        return sol.t_events[0][0] ** 2
    return np.nan


def delta_crit(j):
    th = np.linspace(0.3, 8.0, 240)
    d = np.array([delta_of_theta0(t, j) for t in th])
    ok = np.isfinite(d)
    i = np.argmax(d[ok])
    return d[ok][i], th[ok][i], th[ok], d[ok]


def main():
    print("=" * 100)
    print("FRANK-KAMENETSKII SPATIAL THERMAL-EXPLOSION render→match (spatially resolved Semenov; geometric δ_crit)")
    print("=" * 100)
    res = {}
    for j, (name, lit) in FK_LIT.items():
        dc, t0, ths, ds = delta_crit(j)
        err = abs(dc - lit) / lit
        # confirm the FOLD: δ rises then falls (turning point) — δ at the largest θ₀ is below the peak
        is_fold = ds[-1] < dc - 1e-3
        res[name] = (dc, lit, err, t0, is_fold)
        print(f"  {name:8s} (j={j}): δ_crit(numeric) = {dc:.3f} at θ₀={t0:.2f}  vs FK textbook {lit:.3f}  (err {err*100:.1f}%)  fold={is_fold}")

    slab_ok = res["slab"][2] < 0.05
    ordering = res["slab"][0] < res["cylinder"][0] < res["sphere"][0]
    all_match = all(r[2] < 0.05 for r in res.values())
    all_fold = all(r[4] for r in res.values())
    print(f"\n  ★geometry ordering δ_crit: slab {res['slab'][0]:.2f} < cylinder {res['cylinder'][0]:.2f} < sphere {res['sphere'][0]:.2f} (surface/volume → sphere most stable)")
    print(f"  ★format note: a pouch (slab, δ_crit {res['slab'][0]:.2f}) ignites at {res['cylinder'][0]/res['slab'][0]:.1f}× LOWER reaction strength than a cylindrical cell ({res['cylinder'][0]:.2f})")

    g1 = slab_ok                                            # ★slab δ_crit ≈ 0.878
    g2 = ordering and all_match                             # ★all 3 match + geometric ordering
    g3 = all_fold                                           # ★δ_crit is a fold (turning point)
    g4 = g1 and g2                                          # extends the lumped Semenov criticality to its spatial form
    ok = g1 and g2 and g3
    print("\n" + "-" * 100)
    print(f"  G1 ★slab δ_crit {res['slab'][0]:.3f} ≈ FK 0.878 (err {res['slab'][2]*100:.1f}%)                          {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★geometry δ_crit slab<cyl<sphere, all match FK ({res['cylinder'][0]:.2f}/{res['sphere'][0]:.2f} vs 2.0/3.32)  {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★δ_crit is a saddle-node FOLD (δ(θ₀) turning point) — above it no steady profile (explosion)  {'✓' if g3 else 'FAIL'}")
    print(f"  G4 extends the lumped Semenov criticality into the spatial FK regime (saddle-node)              {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("FRANK-KAMENETSKII SPATIAL THERMAL-EXPLOSION render→match LIVE — the spatially resolved companion to the lumped Semenov")
        print(f"  criticality. Solving the FK boundary-value problem ∇²θ+δe^θ=0 by shooting, the dimensionless reaction strength has a critical FOLD")
        print(f"  value δ_crit that is GEOMETRY-dependent and matches the 1939 textbook to a few percent: slab {res['slab'][0]:.3f} (FK 0.878), cylinder")
        print(f"  {res['cylinder'][0]:.2f} (2.000), sphere {res['sphere'][0]:.2f} (3.322). ★Above δ_crit no steady temperature profile exists — the centre runs away — so")
        print(f"  δ_crit is a saddle-node bifurcation, the criticality primitive now in its SPATIAL form. ★Geometry matters: the sphere is the most")
        print(f"  stable (highest δ_crit, best surface/volume), and a pouch (slab, δ_crit {res['slab'][0]:.2f}) ignites at {res['cylinder'][0]/res['slab'][0]:.1f}× lower reaction strength than a")
        print(f"  cylindrical cell — the cell FORMAT sets the spatial thermal-explosion threshold, the lumped Semenov result at higher fidelity.")
    else:
        print(f"HONEST WIP: g1={g1}(slab {res['slab'][0]:.3f}) g2={g2} g3={g3}. Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
