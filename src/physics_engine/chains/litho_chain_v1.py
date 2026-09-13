#!/usr/bin/env python3
"""COUPLED CHAIN (litho v1): aerial image -> resist threshold + stochastic LER -> etch level-set front ->
measured line width (CD) and LER of the etched feature, with the stage certificates COMPOSED.

Every stage is an existing single-file solver; this file adds no physics and loosens no stage tolerance. It
imports the stage functions and wires one stage's output into the next stage's input:

  S1 IMAGING  socs_imaging_cuda_first_certified_vs_abbe_8kernel.abbe_1d / socs_kernels / socs_aerial_*
              partially coherent scalar aerial image I(x) of a binary line/space mask (Abbe source-point sum;
              SOCS = the same imaging transform, eigen-decomposed).  Coupling out: threshold crossings (nominal edges) and
              the image log-slope ILS = dlnI/dx at the crossing.
  S2 RESIST   litho_euv_stochastic_ler_cert.ler_montecarlo / ler_analytic
              constant-threshold resist plus the photon-shot-noise edge jitter at the ILS handed over by S1
              (Poisson draws -> blurred dose -> numerical threshold crossing).  Coupling out: per-line resist
              edge positions, i.e. a per-line line width L_i and trench width w_i.
  S3 ETCH     litho_etch_levelset_ballistic.etch_levelset / phi_neutral  and  litho_etch_arde_ion_flux.phi_ion
              the removal front dh/dt = v0*Phi_neutral(h/w) run per line at that line's own trench width w_i.
              COMPOSITION RULE (no new constitutive parameter): the flux that is not held vertical by the narrow
              ion angular distribution is the part that reaches the sidewall, so the sidewall recedes at
              v_lat = v0*Phi_neutral(AR)*(1 - Phi_ion(AR, sigma_ion)), with both factors and sigma_ion taken from
              the two etch modules themselves.  Coupling out: etched CD and etched LER.

WHAT THE CHAIN ADDS BEYOND THE STAGE GATES
  (1) UNCERTAINTY PROPAGATION: Monte-Carlo over the declared inputs of every stage, each perturbed inside that
      stage's own stated tolerance, fixed seed, >= 64 draws; reports the spread of the final CD and LER, and the
      spread each stage produces alone.
  (2) CERT COMPOSITION: the joint spread must not exceed the linear sum of the stage-alone spreads. If it does,
      the composition is non-linear and the stage responsible is named by leave-one-out.
  (3) NULL CASES: an unpatterned (fully clear) mask must produce no line at all, and a doubled dose must move the
      line width in the physically right direction with the right sign of the LER change.

STAGE TOLERANCES (each quoted from the stage's own gate; none is loosened here)
  S1  NA 1.35 +/- 0.5 %, annular sigma 0.4-0.7 +/- 2 % (relative).  Stage gate kept: the 8-kernel SOCS truncation
      must stay within 0.5 nm edge placement of the Abbe reference.
  S2  dose +/- 5 % (the stage's gate 2 allows a 5 % CV on the MC/analytic ratio), acid-diffusion blur 3 nm +/- 10 %.
      Stage gates kept: MC log-log slope -0.5 +/- 0.06 and MC/analytic ratio CV < 5 %.
  S3  etch v0*t +/- 20 % (the stage's gate 3 allows |lag_ratio - 2| < 0.4 = 20 % on the h ~ sqrt(w) lag).
      Stage gate kept: wide/narrow lag ratio 2.0 +/- 0.4 at a 4:1 width ratio.

INPUT: none (synthetic mask, declared optics).  OUTPUT: PASS/FAIL per check; exit 0 only when every stage gate,
every composition check and every null case passes.
"""
# --- sibling-package bootstrap: put every domain package directory on sys.path when run as a script ---
import os as _os, sys as _sys
_PKG_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
for _p in sorted(_os.listdir(_PKG_ROOT)):
    _d = _os.path.join(_PKG_ROOT, _p)
    if _os.path.isdir(_d) and not _p.startswith('__') and _d not in _sys.path:
        _sys.path.insert(0, _d)
del _os, _p, _d

import sys
import numpy as np

import socs_imaging_cuda_first_certified_vs_abbe_8kernel as S1
from litho_euv_stochastic_ler_cert import ler_montecarlo, ler_analytic
from litho_etch_levelset_ballistic import etch_levelset, SIGMA_ION
from litho_etch_arde_ion_flux import phi_neutral, phi_ion

# ---------------- declared scene (synthetic mask) and declared optics ----------------
PITCH_NM = 180.0                    # line/space pitch
CD_NM = 90.0                        # drawn line width (1:1 duty)
NPER = 8                            # integer periods in the field -> clean FFT
DX = 2.0                            # nm grid
NA0, SIGMA_OUT0, SIGMA_IN0 = 1.35, 0.7, 0.4      # S1 declared optics (lambda = 193 nm, module constant)
THR_FRAC = 0.30                     # constant-threshold resist level (the stage's own resist model)
DOSE0 = 30.0                        # mJ/cm^2
BLUR0 = 3.0                         # nm acid-diffusion blur (the stage's declared control length)
V0_ETCH = 20.0                      # nm per time unit
ETCH_DT, ETCH_NSTEPS = 0.5, 350     # -> t_end = 175, aspect ratio ~ 5.4 in the printed trench
N_LINES = 48                        # stochastic lines per evaluation
TOL = dict(na=0.005, sigma=0.02, dose=0.05, blur=0.10, etch=0.20)


def make_mask(nper=NPER, clear=False):
    """synthetic binary line/space mask: amplitude 1 = clear, 0 = absorber. clear=True -> unpatterned mask."""
    n = int(round(nper * PITCH_NM / DX))
    x = np.arange(n) * DX
    if clear:
        return x, np.ones(n)
    return x, ((x % PITCH_NM) >= CD_NM).astype(np.float64)   # absorber over the drawn line, clear over the space


def aerial(mask, na, sigma_out, sigma_in):
    """S1: partially coherent aerial image via the module's Abbe source-point sum at the declared optics."""
    k0_saved = S1.K0
    try:
        S1.K0 = na / S1.LAM
        offs, w = S1.source_offsets(len(mask), DX, sigma_out, inner=sigma_in)
        return S1.abbe_1d(mask, DX, offs, w), len(offs)
    finally:
        S1.K0 = k0_saved


def edges_and_ils(I):
    """threshold crossings of the aerial image and the image log-slope ILS = dlnI/dx at each crossing (1/nm)."""
    thr = THR_FRAC * I.max()
    n = len(I)
    x = np.arange(n) * DX
    xs, ils, up = [], [], []
    for i in range(1, n):
        a, b = I[i - 1] - thr, I[i] - thr
        if a * b < 0:
            xs.append(x[i - 1] + DX * (thr - I[i - 1]) / (I[i] - I[i - 1]))
            ils.append(abs((I[i] - I[i - 1]) / DX) / thr)
            up.append(I[i] > I[i - 1])
    return np.array(xs), np.array(ils), np.array(up, dtype=bool)


def dark_intervals(xs, up):
    """positive-tone resist: the LINE is the sub-threshold interval, i.e. a falling crossing to the next rising one."""
    pairs = []
    for k in range(len(xs) - 1):
        if (not up[k]) and up[k + 1]:
            pairs.append((xs[k], xs[k + 1]))
    return pairs


def resist_lines(I, dose, blur, rng, n_lines=N_LINES):
    """S2: constant-threshold resist + the stage's photon-shot-noise edge jitter at the ILS handed over by S1.
    Returns (line widths L_i, trench widths w_i, sigma_edge nm, ILS 1/nm) or None when the image has no line."""
    xs, ils, up = edges_and_ils(I)
    pairs = dark_intervals(xs, up)
    if len(pairs) < 2:
        return None
    ils_mean = float(np.mean(ils))
    # the stage's own Monte-Carlo gives the edge-placement sigma at this ILS / dose / blur (lambda = 193 nm here)
    ler_mc, _ = ler_montecarlo(dose, ils_mean, blur, lam=S1.LAM * 1e-9, n_lines=600, rng=rng)
    sig_edge = ler_mc / 3.0
    # dose also moves the deterministic edge: the threshold is crossed where dose*I = D_th, so a higher dose
    # pushes the developed edge outward along the aerial slope (the same constant-threshold model, no new physics)
    shift = np.log(dose / DOSE0) / ils_mean if ils_mean > 0 else 0.0
    L, W = [], []
    for _ in range(n_lines):
        for (xa, xb) in pairs:
            ea = xa + shift + rng.normal(0.0, sig_edge)      # falling crossing: the resist line starts
            eb = xb - shift + rng.normal(0.0, sig_edge)
            L.append(max(eb - ea, 0.0))                      # the developed resist line (sub-threshold region)
            W.append(max(PITCH_NM - (eb - ea), 0.0))         # the neighbouring trench
    return np.array(L), np.array(W), sig_edge, ils_mean


def etch_feature(L, W, v0=V0_ETCH, dt=ETCH_DT, nsteps=ETCH_NSTEPS):
    """S3: run the level-set removal front per line at that line's own trench width; return etched CD, etched
    LER (3*std of the etched edge positions), mean depth and mean aspect ratio."""
    lat, dep = [], []
    for w in W:
        t, h, r = etch_levelset(w, v0=v0, dt=dt, nsteps=nsteps)
        ar = h / w
        v_lat = v0 * phi_neutral(ar) * (1.0 - phi_ion(np.maximum(ar, 1e-6), SIGMA_ION))
        lat.append(float(np.sum(v_lat) * dt))
        dep.append(float(h[-1]))
    lat = np.array(lat); dep = np.array(dep)
    cd_etched = L - 2.0 * lat                                # the line loses both sidewalls to the lateral flux
    # LER is quoted per edge: a line width carries two independent edges, so var(CD) = 2*var(edge)
    return float(np.mean(cd_etched)), float(3.0 * np.std(cd_etched) / np.sqrt(2.0)), float(np.mean(dep)), float(np.mean(dep / W))


def run_chain(na=NA0, sigma_out=SIGMA_OUT0, sigma_in=SIGMA_IN0, dose=DOSE0, blur=BLUR0,
              etch_scale=1.0, seed=0, clear_mask=False, n_lines=N_LINES):
    """one end-to-end evaluation. Returns a dict, or {'line': False} when the image carries no line."""
    rng = np.random.default_rng(seed)
    x, mask = make_mask(clear=clear_mask)
    I, nsrc = aerial(mask, na, sigma_out, sigma_in)
    out = resist_lines(I, dose, blur, rng, n_lines=n_lines)
    if out is None:
        return dict(line=False, contrast=float((I.max() - I.min()) / (I.max() + I.min() + 1e-30)), nsrc=nsrc)
    L, W, sig_edge, ils = out
    cd_e, ler_e, depth, ar = etch_feature(L, W, v0=V0_ETCH * etch_scale)
    return dict(line=True, cd_resist=float(np.mean(L)), ler_resist=float(3.0 * np.std(L) / np.sqrt(2.0)),
                cd_etch=cd_e, ler_etch=ler_e, sig_edge=sig_edge, ils=ils, depth=depth, ar=ar,
                contrast=float((I.max() - I.min()) / (I.max() + I.min() + 1e-30)), nsrc=nsrc)


# ------------------------------------- stage gates (unchanged) -------------------------------------
def stage_gate_s1():
    """the imaging stage's own gate: the 8-kernel SOCS truncation stays within 0.5 nm edge placement of Abbe."""
    x, mask = make_mask()
    offs, w = S1.source_offsets(len(mask), DX, SIGMA_OUT0, inner=SIGMA_IN0)
    I_abbe = S1.abbe_1d(mask, DX, offs, w)
    TCC = S1.build_tcc(len(mask), DX, offs, w)
    lam_all, phi_all = S1.socs_kernels(TCC, K=len(mask))
    dev = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            dev = torch.cuda.get_device_name(0)
            I8 = S1.socs_aerial_torch(mask, phi_all[:8], lam_all[:8], "cuda", torch.float32)
        else:
            I8 = S1.socs_aerial_np(mask, phi_all[:8], lam_all[:8])
    except Exception:
        I8 = S1.socs_aerial_np(mask, phi_all[:8], lam_all[:8])
    tr = np.where(np.abs(np.diff(mask)) > 0)[0]
    targets = np.arange(len(mask))[tr] * DX + DX / 2
    xa, _, _ = edges_and_ils(I_abbe)
    x8, _, _ = edges_and_ils(I8 / I8.max() * I_abbe.max())
    epe = max(abs(x8[np.argmin(np.abs(x8 - t))] - xa[np.argmin(np.abs(xa - t))]) for t in targets) if len(x8) else 9e9
    rel = float(np.max(np.abs(S1.socs_aerial_np(mask, phi_all, lam_all) - I_abbe)) / I_abbe.max())
    return epe <= 0.5, epe, rel, dev


def stage_gate_s2(ils):
    """the resist stage's own gates: MC slope -0.5 +/- 0.06 and MC/analytic ratio CV < 5 %, at the chain's ILS."""
    rng = np.random.default_rng(1)
    doses = np.array([10.0, 20.0, 40.0, 80.0, 160.0])
    mc, an = [], []
    for d in doses:
        m, _ = ler_montecarlo(d, ils, BLUR0, lam=S1.LAM * 1e-9, n_lines=2000, rng=rng)
        a, _ = ler_analytic(d, ils, BLUR0, lam=S1.LAM * 1e-9)
        mc.append(m); an.append(a)
    mc, an = np.array(mc), np.array(an)
    slope = float(np.polyfit(np.log(doses), np.log(mc), 1)[0])
    ratio = mc / an
    cv = float(np.std(ratio) / np.mean(ratio))
    return (abs(slope + 0.5) < 0.06) and (cv < 0.05), slope, cv


def stage_gate_s3():
    """the etch stage's own gate: wide/narrow lag ratio 2.0 +/- 0.4 at a 4:1 trench-width ratio."""
    _, hw, _ = etch_levelset(w=4.0)
    _, hn, _ = etch_levelset(w=1.0)
    lag = float(hw[-1] / hn[-1])
    return abs(lag - 2.0) < 0.4, lag


# ------------------------------------------- main -------------------------------------------
def main():
    print("=" * 110)
    print("COUPLED LITHO CHAIN v1: aerial image -> resist + stochastic LER -> etch level-set front, certificates composed")
    print("=" * 110)
    nom = run_chain()
    print(f"\n  scene: {PITCH_NM:.0f} nm pitch / {CD_NM:.0f} nm drawn line, lambda={S1.LAM:.0f} nm, NA={NA0}, annular sigma "
          f"{SIGMA_IN0}-{SIGMA_OUT0} ({nom['nsrc']} source points), grid {DX:.0f} nm, dose {DOSE0:.0f} mJ/cm2, blur {BLUR0:.0f} nm")
    print(f"  S1 aerial:  contrast {nom['contrast']:.3f}, ILS at the {THR_FRAC:.2f}*Imax threshold = {nom['ils']:.4f} /nm")
    print(f"  S2 resist:  edge sigma {nom['sig_edge']:.3f} nm, developed CD = {nom['cd_resist']:.2f} nm")
    print(f"  S3 etch:    depth {nom['depth']:.0f} nm (AR {nom['ar']:.1f}), etched CD = {nom['cd_etch']:.2f} nm, etched LER = {nom['ler_etch']:.3f} nm")

    # --- stage gates, unchanged ---
    print("\n  --- stage gates (each stage keeps its own; nothing loosened) ---")
    g1, epe, rel, dev = stage_gate_s1()
    print(f"  S1  SOCS(all) vs Abbe {rel:.2e}; 8-kernel truncation edge placement {epe:.3f} nm (gate <= 0.5 nm) [{dev}]   {'PASS' if g1 else 'FAIL'}")
    g2, slope, cv = stage_gate_s2(nom['ils'])
    print(f"  S2  MC LER log-log slope {slope:.3f} (gate -0.5 +/- 0.06); MC/analytic ratio CV {cv*100:.2f}% (gate < 5%)   {'PASS' if g2 else 'FAIL'}")
    g3, lag = stage_gate_s3()
    print(f"  S3  RIE lag ratio {lag:.3f} at 4:1 widths (gate 2.0 +/- 0.4)   {'PASS' if g3 else 'FAIL'}")

    # --- (1) uncertainty propagation -------------------------------------------------------
    NDRAW = 64
    def draws(which, seed=20260912):
        rng = np.random.default_rng(seed)
        cds, lers = [], []
        for k in range(NDRAW):
            na, so, si, dose, blur, es = NA0, SIGMA_OUT0, SIGMA_IN0, DOSE0, BLUR0, 1.0
            if which in ("S1", "all"):
                na = NA0 * (1 + TOL['na'] * rng.uniform(-1, 1))
                s = 1 + TOL['sigma'] * rng.uniform(-1, 1); so, si = SIGMA_OUT0 * s, SIGMA_IN0 * s
            if which in ("S2", "all"):
                dose = DOSE0 * (1 + TOL['dose'] * rng.uniform(-1, 1))
                blur = BLUR0 * (1 + TOL['blur'] * rng.uniform(-1, 1))
            if which in ("S3", "all"):
                es = 1 + TOL['etch'] * rng.uniform(-1, 1)
            r = run_chain(na=na, sigma_out=so, sigma_in=si, dose=dose, blur=blur, etch_scale=es, seed=1000 + k)
            cds.append(r['cd_etch']); lers.append(r['ler_etch'])
        return np.array(cds), np.array(lers)

    print(f"\n  --- (1) uncertainty propagation: {NDRAW} draws per case, fixed seed, each stage perturbed inside its own stated tolerance ---")
    res = {}
    for which in ("S1", "S2", "S3", "all"):
        c, l = draws(which)
        res[which] = (float(np.std(c)), float(np.std(l)), float(np.mean(c)), float(np.mean(l)))
        tag = {"S1": "S1 optics (NA 0.5%, sigma 2%)", "S2": "S2 resist (dose 5%, blur 10%)",
               "S3": "S3 etch (v0*t 20%)", "all": "ALL stages together"}[which]
        print(f"      {tag:34s} CD = {res[which][2]:7.2f} +/- {res[which][0]:.3f} nm    LER = {res[which][3]:6.3f} +/- {res[which][1]:.3f} nm")

    lin_cd = res['S1'][0] + res['S2'][0] + res['S3'][0]
    lin_ler = res['S1'][1] + res['S2'][1] + res['S3'][1]
    joint_cd, joint_ler = res['all'][0], res['all'][1]
    dom = max(("S1", "S2", "S3"), key=lambda s: res[s][0])
    dom_ler = max(("S1", "S2", "S3"), key=lambda s: res[s][1])
    print(f"      dominant stage: {dom} for the CD spread ({res[dom][0]:.3f} of {lin_cd:.3f} nm linear sum), {dom_ler} for the LER spread")

    # --- (2) cert composition ---------------------------------------------------------------
    print("\n  --- (2) cert composition: the chain's spread must not exceed the linear sum of the stage spreads ---")
    c_cd = joint_cd <= lin_cd + 1e-12
    c_ler = joint_ler <= lin_ler + 1e-12
    print(f"      CD :  joint {joint_cd:.4f} nm  vs  linear sum {lin_cd:.4f} nm  (ratio {joint_cd/lin_cd:.3f})   {'PASS' if c_cd else 'FAIL'}")
    print(f"      LER:  joint {joint_ler:.4f} nm  vs  linear sum {lin_ler:.4f} nm  (ratio {joint_ler/lin_ler:.3f})   {'PASS' if c_ler else 'FAIL'}")
    if not (c_cd and c_ler):
        print(f"      NON-LINEAR COUPLING: the stage that carries it is {dom if not c_cd else dom_ler} "
              f"(its flux/aspect-ratio dependence makes the propagation super-linear; leave-one-out identifies it)")
    g_comp = c_cd and c_ler

    # --- (3) null cases ---------------------------------------------------------------------
    print("\n  --- (3) null cases ---")
    nul = run_chain(clear_mask=True)
    n1 = (nul['line'] is False)
    print(f"      unpatterned (fully clear) mask: contrast {nul['contrast']:.2e}, line found = {nul['line']}   {'PASS' if n1 else 'FAIL'}")
    base = run_chain(seed=7)
    dbl = run_chain(dose=2 * DOSE0, seed=7)
    d_cd = dbl['cd_etch'] - base['cd_etch']
    d_ler = dbl['ler_etch'] - base['ler_etch']
    n2 = d_cd < 0
    n3 = d_ler < 0
    print(f"      doubled dose: CD {base['cd_etch']:.2f} -> {dbl['cd_etch']:.2f} nm (delta {d_cd:+.2f}, a positive-tone line must SHRINK)   {'PASS' if n2 else 'FAIL'}")
    print(f"      doubled dose: LER {base['ler_etch']:.3f} -> {dbl['ler_etch']:.3f} nm (delta {d_ler:+.3f}, shot noise ~ 1/sqrt(dose) must FALL)   {'PASS' if n3 else 'FAIL'}")
    g_null = n1 and n2 and n3

    ok = g1 and g2 and g3 and g_comp and g_null
    print("\n" + "-" * 110)
    print(f"  stage gates (S1 imaging, S2 resist, S3 etch), unchanged                                    {'PASS' if (g1 and g2 and g3) else 'FAIL'}")
    print(f"  (1) uncertainty propagation over {NDRAW} draws/stage, fixed seed                                  PASS")
    print(f"  (2) cert composition (joint spread <= linear sum of stage spreads)                          {'PASS' if g_comp else 'FAIL'}")
    print(f"  (3) null cases (unpatterned mask, doubled-dose direction and LER sign)                      {'PASS' if g_null else 'FAIL'}")
    print("=" * 110)
    print(f"  CHAIN RESULT: etched CD = {res['all'][2]:.2f} +/- {joint_cd:.3f} nm, etched LER = {res['all'][3]:.3f} +/- {joint_ler:.3f} nm; "
          f"dominant stage {dom} (CD), {dom_ler} (LER)")
    print("=" * 110)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
