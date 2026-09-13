"""BETHE-BLOCH STOPPING — A CHARGED PARTICLE DUMPS ITS ENERGY AT THE END OF ITS RANGE (particle-matter
interaction) — RENDER->MATCH how fast a charged particle loses energy plowing through matter, and WHY it deposits most of that energy
right before it stops -- the Bragg peak that makes proton therapy possible. Each atomic electron the particle passes gets a Coulomb kick
delta E ~ 1/b^2; summing over all impact parameters b from a close-collision minimum to an adiabatic maximum gives the stopping power
-dE/dx = (4 pi n Z^2 e^4/m_e v^2) ln(b_max/b_min) -- rising as 1/v^2 as the particle SLOWS. We do NOT assert it: we integrate the genuine
energy-transfer over impact parameters. ★The genuine integral equals the Bohr/Bethe stopping formula; ★because -dE/dx ~ 1/v^2, a particle
deposits little energy while fast and a BURST right before it stops -- integrating a proton's slowing-down gives the BRAGG PEAK, several
times the entrance dose at the end of the range (the cross-check); ★a faster particle ionises less (1/v^2, the null); ★doubling the charge
quadruples the stopping (Z^2). The particle speed v is the physical sigma. Gates run through render_match_scaffold.

MATCH: the genuine impact-parameter energy-transfer integral equals (4 pi n Z^2 e^4/m_e v^2) ln(b_max/b_min); ★the 1/v^2 rise makes the Bragg peak at the end of the range and the stopping scales as Z^2 (cross-checks); ★a fast particle ionises little (the null).
  python quantum/bethe_bloch.py
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
import os
import numpy as np
from scipy.integrate import quad

sys.path.insert(0, os.path.dirname(__file__))
from render_match_scaffold import Benchmark, render_match

Z0 = 1.0           # projectile charge
NE = 1.0           # electron density (natural units)
W0 = 1.0           # mean electron orbital frequency (ionisation)
V_OP = 5.0         # projectile speed


def stopping(v=V_OP, Z=Z0):
    """genuine: energy-transfer integral over impact parameters b (close-collision to adiabatic cutoff)."""
    bmin = Z / v ** 2; bmax = v / W0                                # b_min ~ Ze^2/(m_e v^2), b_max ~ v/omega_0
    if bmax <= bmin:
        return 0.0
    return float(quad(lambda b: (2 * Z ** 2 / (v ** 2 * b ** 2)) * NE * 2 * np.pi * b, bmin, bmax)[0])


def stopping_analytic(v=V_OP, Z=Z0):
    bmin = Z / v ** 2; bmax = v / W0
    return float(4 * np.pi * NE * Z ** 2 / v ** 2 * np.log(bmax / bmin)) if bmax > bmin else 0.0


def bragg_profile(v0=8.0, M=1836.0, dx=2.0):
    """integrate a projectile slowing down; return the dose (dE/dx) profile vs depth."""
    E = 0.5 * M * v0 ** 2; x = 0.0; xs = []; dose = []
    while x < 1e7:
        v = np.sqrt(2 * E / M); S = stopping(v)
        if S <= 0 or v < 1.7:
            break
        xs.append(x); dose.append(S); E -= S * dx; x += dx
    return np.array(xs), np.array(dose)


def main():
    print("=" * 96)
    print("BETHE-BLOCH STOPPING — a charged particle dumps its energy at the end of its range; render->match")
    print("=" * 96)
    def rfn(p):
        return stopping(v=p.get("v", V_OP), Z=p.get("Z", Z0))
    band = [{"v": 0.92 * V_OP}, {"v": 1.08 * V_OP}]                # sigma = projectile speed v
    bench = stopping_analytic(V_OP)
    res = render_match(
        rfn, band, {"v": V_OP},
        Benchmark("stopping power -dE/dx", bench, 0.0, "(4 pi n Z^2 e^4/m_e v^2) ln(b_max/b_min) (EXTERNAL: Bethe-Bloch)", ""),
        nulls=[("a much FASTER particle (v = 3 v_op) ionises far LESS: the stopping power falls as 1/v^2, so tripling the speed cuts the energy loss to roughly a ninth. A fast particle spends little time near each electron and barely kicks it; this is why minimum-ionising particles streak through detectors leaving only a faint track, and why the energy is dumped only once the particle has SLOWED (fast -> little stopping)", {"v": 3 * V_OP}, lambda v, m: v < 0.3 * bench)],
        perturbations=[("DOUBLING the projectile charge (Z = 2, e.g. an alpha vs a proton) roughly QUADRUPLES the stopping power -- it scales as Z^2 because both the force on each electron and the number of contributing collisions grow with the charge, softened slightly to ~3.4x by the ln(v^3/Z) term (the close-collision cutoff b_min ~ Z grows with charge, trimming the log). Heavier, more-charged ions stop far more sharply and deposit their energy in a shorter range; this near-Z^2 law is why alpha particles are stopped by paper while protons need centimetres (Z=2 -> ~3.4x stopping)", {"Z": 2.0}, lambda v, best: v > 3 * best)],
        notes=["the genuine impact-parameter integral equals the Bethe-Bloch formula; the 1/v^2 rise makes the Bragg peak; stopping ~ Z^2; a fast particle ionises little"])
    print(res.report())
    s = stopping(); xs, dose = bragg_profile(); ip = int(np.argmax(dose)); bragg_ratio = dose[ip] / dose[0]
    sv2 = {v: stopping(v) * v ** 2 for v in (3, 5, 8)}; s_fast = stopping(3 * V_OP); s_z2 = stopping(Z=2.0)
    print(f"\n  ★STOPPING = IMPACT-PARAMETER ENERGY-TRANSFER INTEGRAL (computed, not asserted): summing the Coulomb energy kick delta E ~ 1/b^2 over all impact parameters from b_min to b_max gives -dE/dx = {s:.4f} vs (4 pi n Z^2/v^2) ln(b_max/b_min) = {bench:.4f} ({abs(s-bench)/bench*100:.2f}%). The stopping is built from the collisions, not asserted; the 1/v^2 prefactor shows as S*v^2 = " + ", ".join(f"{v}:{p:.1f}" for v, p in sv2.items()) + " (rising only slowly through the log)")
    print(f"  ★★THE BRAGG PEAK (the falsifier -- why proton therapy works): integrating a proton's slowing-down, the dose dE/dx is LOW at the entrance and spikes to {bragg_ratio:.1f}x that just before the particle stops, at {xs[ip]/xs[-1]*100:.0f}% of the range. Because -dE/dx ~ 1/v^2 blows up as v -> 0, the energy is dumped in a sharp peak at a controllable DEPTH -- a proton beam can be tuned to deposit its dose right on a tumour and spare the tissue in front, unlike an exponentially-attenuating photon beam")
    print(f"  ★1/v^2 AND Z^2 (cross-check / the null): a particle 3x faster loses only {s_fast:.4f} ({s_fast/s*100:.0f}% of the stopping) -- fast particles are minimum-ionising. Double the charge and the stopping jumps to {s_z2:.4f} = {s_z2/s:.1f}x (~Z^2 = 4x, softened by the ln(v^3/Z) term), which is why an alpha stops in paper and a proton needs centimetres. Speed and charge set where and how sharply a particle stops")
    print(f"  (4) ★WHY IT MATTERS: the Bethe-Bloch stopping power is the foundation of RADIATION DOSIMETRY and hadron therapy (the Bragg peak places the dose on the tumour), of particle detectors (dE/dx identifies particles), of radiation shielding, and of single-event-upset rates in electronics. -dE/dx(v) and the Bragg curve are what a radiation / therapy / detector twin validates against")
    g4 = res.ok and abs(s - bench) / bench < 0.005 and bragg_ratio > 3.0 and ip > 0.7 * len(dose) and sv2[8] > sv2[3]  # integral=Bethe; Bragg peak near the end; slow-log rise
    g5 = s_fast < 0.3 * bench and 3.0 < s_z2 / s < 4.0                                                                # fast null; near-Z^2 quadrupling (softened by the ln(v^3/Z) term)
    ok = g4 and g5
    import json
    os.makedirs("artifacts", exist_ok=True)
    with open("artifacts/bethe_bloch.json", "w") as fh:
        json.dump({"module": "bethe_bloch", "provenance": "self-contained: impact-parameter energy-transfer integral vs Bethe-Bloch; Bragg-peak from integrating the slowing-down",
                   "Z": Z0, "v_op": V_OP, "stopping": s, "stopping_analytic": bench, "bragg_ratio": float(bragg_ratio), "bragg_depth_frac": float(xs[ip] / xs[-1]),
                   "stopping_fast": s_fast, "stopping_Z2": s_z2, "Z2_ratio": s_z2 / s,
                   "render_match_ok": bool(res.ok), "band": [float(res.band_lo), float(res.band_hi)],
                   "cross_checks": {"integral_bragg": bool(g4), "null_fast_Z2": bool(g5)},
                   "all_pass": bool(ok)}, fh, indent=2)
    print("\n" + "=" * 96)
    if ok:
        print("RENDER→MATCH CLOSES (Bethe-Bloch stopping) — a charged particle dumps its energy at the end of its range:")
        print(f"  • -dE/dx = {s:.4f} = (4 pi n Z^2/v^2) ln(b_max/b_min) (band=speed σ); genuine impact-parameter integral.")
        print(f"  • ★Bragg peak {bragg_ratio:.1f}x at {xs[ip]/xs[-1]*100:.0f}% of range (proton therapy); 1/v^2 (fast=null); Z^2 (alpha vs proton).")
        print(f"  • ★a distinct particle-matter/dosimetry primitive -- the Bethe-Bloch closure; distinct from Rutherford + Thomson.")
    else:
        print(f"  HONEST: scaffold ok={res.ok}, integral/bragg {g4}, null/Z2 {g5}. Fix at source.")
    print("=" * 96)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
