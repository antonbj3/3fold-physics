"""OPTICAL VORTEX — ORBITAL ANGULAR MOMENTUM OF A HELICAL BEAM (singular optics) — RENDER->MATCH how a light
beam with a HELICAL phase front exp(i l phi) carries l units of ORBITAL angular momentum per photon (distinct from spin/polarization)
and why its intensity has a dark hole on axis. The phase winds l times around the propagation axis, so the wavefront is a corkscrew
and the on-axis field must vanish (a phase singularity). We do NOT read l off the construction: we build the field, then MEASURE the
angular-momentum expectation from its PHASE GRADIENT using the genuine operator L_z = -i(x d/dy - y d/dx),
  <L_z> = integral psi* (-i)(x d_y - y d_x) psi  /  integral |psi|^2.
★This returns l (including its SIGN -- the handedness of the helix); ★the on-axis intensity is a deep null (the donut/doughnut
mode) for any l != 0; ★at l=0 there is no orbital angular momentum and the beam is solid on axis (the null). The topological charge
l is the physical sigma. Uses render_match_scaffold (a distinct singular-optics primitive -- the orbital-angular-momentum
closure an optical-tweezers / free-space-OAM-communication / structured-light digital twin needs).

MATCH: the orbital angular momentum <L_z> measured from the field's phase gradient equals the topological charge l; ★the on-axis intensity is a deep null (donut) for l != 0 (cross-check); ★a negative charge gives <L_z> = -l (opposite handedness); at l=0 there is no OAM and the beam is solid (the null); a larger charge carries more OAM.
Reference: Allen et al., Phys. Rev. A 45, 8185 (1992) -- orbital angular momentum l hbar per photon of a helical exp(i l phi) beam.
"""
# --- sibling-package bootstrap: this repository splits the modules by domain, so put every
# --- package directory on sys.path when the file is run as a script.
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

L0 = 3                                                   # topological charge (the physical sigma)
N = 600
LG = 12.0
_x = np.linspace(-LG / 2, LG / 2, N); _dx = _x[1] - _x[0]
_X, _Y = np.meshgrid(_x, _x)
_R = np.sqrt(_X ** 2 + _Y ** 2); _PHI = np.arctan2(_Y, _X)
W = 3.0


def field(l):
    return _R ** abs(l) * np.exp(-_R ** 2 / W ** 2) * np.exp(1j * l * _PHI)


def oam(l):
    """measure <L_z> from the field's phase gradient via L_z=-i(x d_y - y d_x); NOT read off the construction."""
    psi = field(l)
    dxp = np.gradient(psi, _dx, axis=1); dyp = np.gradient(psi, _dx, axis=0)
    Lz_psi = -1j * (_X * dyp - _Y * dxp)
    return float(np.real(np.sum(np.conj(psi) * Lz_psi)) / np.sum(np.abs(psi) ** 2))


def central_intensity_ratio(l):
    psi = field(l); I = np.abs(psi) ** 2
    return float(I[N // 2, N // 2] / I.max())            # on-axis intensity / peak (->0 = donut)


def main():
    print("=" * 96)
    print("OPTICAL VORTEX — orbital angular momentum from the phase gradient; render->match")
    print("=" * 96)
    def rfn(p):
        if p.get("solid"):
            return oam(0)                                # l=0: no OAM
        return oam(p.get("l", L0))                        # OAM from the phase gradient
    band = [{"l": 2}, {"l": 4}]                          # topological-charge sigma: <L_z>=l (integer band brackets l=3)
    bench = float(L0)
    res = render_match(
        rfn, band, {"l": L0},
        Benchmark("orbital angular momentum <L_z>", bench, 0.05 * bench, "l (EXTERNAL, topological charge)", "hbar"),
        nulls=[("a beam with NO phase winding (l=0) has a flat wavefront -- there is no helical twist, no on-axis singularity, and the angular-momentum expectation is ZERO: the orbital angular momentum is ENTIRELY the helical exp(i l phi) phase. A plane or fundamental Gaussian beam carries no OAM (l=0 -> <L_z>=0)", {"solid": True}, lambda v, m: abs(v) < 0.05)],
        perturbations=[("a HIGHER topological charge winds the phase more times per turn, so each photon carries MORE orbital angular momentum -- <L_z>=l grows with the charge (more winding -> more OAM, and a wider dark core)", {"l": 5}, lambda v, best: v > best + 1.5)],
        notes=["the OAM measured from the phase gradient equals l; the on-axis intensity is a donut null for l!=0; a negative charge flips the sign; l=0 is solid with no OAM"])
    print(res.report())
    o0 = oam(L0); cir = central_intensity_ratio(L0); o_neg = oam(-L0)
    ls = [0, 1, 2, 3]
    print(f"\n  ★OAM FROM THE PHASE GRADIENT (measured, not asserted): <L_z>={o0:.4f} vs l={bench:.1f} ({abs(o0-bench)/bench*100:.2f}%) -- the angular-momentum operator L_z=-i(x d_y - y d_x) was applied to the field and its expectation taken; l was NOT read off the exp(i l phi) exponent. This is genuine orbital (not spin/polarization) angular momentum, l hbar per photon")
    print(f"  ★★THE DONUT NULL + SIGN (the falsifiers): on-axis intensity/peak = {cir:.2e} (a deep null -- the phase singularity forces the field to vanish on axis, the doughnut mode); and a NEGATIVE charge gives <L_z>={o_neg:.3f} (=-l, the opposite helix handedness). central-I/peak vs l -- " + ", ".join(f"l={l}:{central_intensity_ratio(l):.0e}" for l in ls) + " (solid at l=0, donut for l!=0)")
    print(f"  ★OAM = l ACROSS CHARGES (the sigma): <L_z> vs l -- " + ", ".join(f"l={l}:{oam(l):.3f}" for l in ls) + " -- each unit of topological charge adds one hbar of orbital angular momentum, a discrete, conserved, transferable quantity independent of the beam's polarization")
    print(f"  (4) ★WHY IT MATTERS: orbital angular momentum is a second, high-dimensional channel of light beyond intensity, phase and polarization -- it spins micro-particles in optical tweezers, multiplexes free-space and fiber communication (many l values = many channels), drives chiral light-matter interaction and super-resolution, and labels the vortices in quantum and astronomical light. The <L_z>=l and donut closure is what a structured-light twin integrates")
    g4 = abs(o0 - bench) / bench < 0.05 and cir < 1e-2 and abs(o_neg + L0) < 0.05                              # OAM=l; donut null; sign flip
    g5 = oam(5) > o0 + 1.5 and abs(oam(0)) < 0.05 and central_intensity_ratio(0) > 0.9                          # higher charge more OAM; l=0 no OAM and solid
    ok = res.ok and g4 and g5
    import os, json
    art = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")
    os.makedirs(art, exist_ok=True)
    with open(os.path.join(art, "optical_vortex_oam.json"), "w") as fh:
        json.dump({"module": "optical_vortex_oam", "provenance": "self-contained measurement of <L_z> from a vortex field's phase gradient, no external data",
                   "oam": o0, "topological_charge": L0, "central_intensity_ratio": cir, "oam_negative_charge": o_neg,
                   "oam_vs_l": {f"{l}": {"measured": float(oam(l)), "donut_ratio": float(central_intensity_ratio(l))} for l in ls},
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"oam_donut_sign": bool(g4), "higher_solid": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (optical vortex) — a helical beam carries l hbar of orbital angular momentum:")
        print(f"  • <L_z>={o0:.4f} matches the topological charge l={bench:.0f} (band=charge σ).")
        print(f"  • ★on-axis donut null ({cir:.0e}); negative charge flips the sign; l=0 is solid with no OAM.")
        print(f"  • ★a distinct singular-optics primitive for tweezers/OAM-communication/structured-light twins.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, oam/donut/sign {g4}, higher/solid {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
