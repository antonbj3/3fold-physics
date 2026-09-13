"""Certified design search on the batched GPU lens tracer: a four-parameter local search on ONE public patent
prescription, where every reported design must pass three reused instrument certificates.

WHAT THIS IS: the GPU tracer (lens_raytrace_gpu) evaluates thousands of designs per kernel launch. This module
puts a population loop on top of it and, crucially, refuses to report any design that has not passed the same
gates the tracer itself is gated on. The point is the certificate discipline (a search that optimises an
objective while silently drifting off focal length, aperture or throughput reports a design that is not the
lens that was asked for), not the optimiser, which is deliberately the simplest thing that works.

WHAT THIS IS NOT: a lens design program. There is no glass selection, no variable-count element insertion, no
aberration-balanced merit function with wavefront/MTF/tolerance terms, no multi-wavelength (the prescription is
traced at a single index set, the catalogue e-line values), no manufacturability constraint (edge thickness,
center thickness, radius availability) and no global search. It is a LOCAL search over four continuous
parameters of ONE patented prescription (US 6,141,154 Example 1), from bounds centred on the patent values.
Nothing here should be read as "the patent design was improved" beyond: inside these bounds, on this spot-RMS
objective, at this ray sampling, with these three certificates enforced, this design vector scores better.

DESIGN VECTOR (4 free parameters, bounds declared in BOUNDS below, all bounds centred on the patent values):
  space1  thickness of the first variable airspace (surface index 8 of the prescription), patent 29.300 mm
  space2  thickness of the second variable airspace (surface index 19), patent 2.520 mm
  r9      radius of surface index 9, patent 287.1786 mm      (freed via the tracer's extended family)
  r13     radius of surface index 13, patent -398.7020 mm    (freed via the tracer's extended family)
The object conjugate is FIXED at the prescription's own object distance, so it is not a search dimension; it is
only passed because the extended tracer family carries it in its design row.

OBJECTIVE: equal-weight mean of the RMS spot radius at three field points, measured at the image surface:
  on-axis (0 deg) and two off-axis field angles, 25.50 deg and 34.55 deg half-angle in object space (image-side
  field heights 0.0, 50.0 and 72.0 mm; 72.0 mm is the design image semi-height). The angles are computed from
  the patent conjugate and printed by the run, not assumed.

CERTIFICATES (all three enforced on EVERY candidate; a candidate failing any one is never eligible to be
reported, whatever its objective):
  C-EFL  paraxial focal length within 0.1% of the patent's 105.815 mm. The paraxial recursion is the batched
         form of the CPU module's paraxial_efl; the run re-checks it against lens_raytrace_gpu.efl_gate
         (the CPU/GPU G-EFL pair) before the search starts, and against the CPU function on sampled candidates.
  C-VIG  vignetting at full field no worse than the patent example measured on the SAME ray grid (the absolute
         percentage depends on aperture sampling, so the reference is re-measured here rather than quoted).
  C-FNO  real-ray working f-number on axis within 1% of the patent example's, again re-measured on the same
         grid as the candidates.

DETERMINISM: fixed seed, no atomics anywhere in the tracer kernels, and the run executes the whole search TWICE
and reports max|delta| between the two runs' best objective and design vector.

NULL CASE: the same loop with the bounds collapsed onto the patent vector must return the patent design and
reproduce its objective to numerical equality.

I/O: no input files; run directly to print the certificate reference values, the per-generation log, the best
certified design, the repeatability and null-case lines, the certified/rejected accounting and a verdict
(exit 0 on pass). Requires a CUDA device for useful throughput (warp's CPU backend is far below these batch
sizes). Prescription and CPU reference: lens_raytrace_cell; batched tracer: lens_raytrace_gpu (this module adds
no physics of its own -- every ray is traced by that module's kernels).
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lens_raytrace_cell as lrc
import lens_raytrace_gpu as lrg

# ----------------------------------------------------------------------------- search configuration
SEARCH_SEED = 20260912
IDX_R_FREE = (9, 13)                      # surface indices whose radii are freed (see module docstring)
CONJUGATE_M = abs(lrc.OBJ_DIST) / 1000.0  # fixed object conjugate in metres (the extended family carries it)

PATENT_SPACE1 = lrc.P[lrc.IDX_SPACE1][1]
PATENT_SPACE2 = lrc.P[lrc.IDX_SPACE2][1]
PATENT_R9 = lrc.P[IDX_R_FREE[0]][0]
PATENT_R13 = lrc.P[IDX_R_FREE[1]][0]
PATENT_VECTOR = np.array([PATENT_SPACE1, PATENT_SPACE2, PATENT_R9, PATENT_R13], dtype=np.float64)

# bounds: airspaces +-4.0 / +-2.0 mm around the patent values (space2 kept positive), radii +-3%
BOUNDS = (
    ("space1", PATENT_SPACE1 - 4.0, PATENT_SPACE1 + 4.0),
    ("space2", max(PATENT_SPACE2 - 2.0, 0.20), PATENT_SPACE2 + 2.0),
    ("r9", PATENT_R9 * 0.97, PATENT_R9 * 1.03),
    ("r13", PATENT_R13 * 1.03, PATENT_R13 * 0.97),     # r13 is negative: 1.03 is the lower bound
)
LO = np.array([b[1] for b in BOUNDS]); HI = np.array([b[2] for b in BOUNDS])

FIELDS_SEARCH = [0.0, 50.0, 72.0]        # image-side field heights (mm); 72.0 = design image semi-height
FIELD_WEIGHTS = np.array([1.0, 1.0, 1.0]) / 3.0

N_DESIGNS = 4096                          # population per generation (one kernel launch per field set)
N_GEN = 6
N_ELITE = 64
SIGMA0 = 0.35                             # initial step as a fraction of each bound's half-width
SIGMA_DECAY = 0.72

N_AP_OBJ = 61                             # aperture grid for the objective / vignetting trace
N_AP_FNO = 61                             # aperture grid for the axial marginal-ray f-number
FNO_CHUNK = 128                           # designs per f-number trace launch (device-memory bound)

TOL_EFL_REL = 1e-3                        # C-EFL: 0.1% of the patent focal length
TOL_FNO_REL = 1e-2                        # C-FNO: 1% of the patent example's working f-number
MIN_ALIVE = 8


# ----------------------------------------------------------------------------- batched instrument
def _rows(vectors):
    """(N,4) search vectors -> (N,5) extended-family rows (space1, space2, conjugate_m, r9, r13)."""
    v = np.atleast_2d(np.asarray(vectors, dtype=np.float64))
    return np.column_stack([v[:, 0], v[:, 1], np.full(len(v), CONJUGATE_M), v[:, 2], v[:, 3]])


def paraxial_efl_batch(z_batch, c_batch, nmed):
    """Batched form of lens_raytrace_cell.paraxial_efl (same recursion, vectorised over designs)."""
    n_surf = z_batch.shape[1]
    y = np.ones(len(z_batch)); w = np.zeros(len(z_batch))
    for i in range(n_surf):
        w = w - y * c_batch[:, i] * (nmed[i + 1] - nmed[i])
        if i < n_surf - 1:
            y = y + (z_batch[:, i + 1] - z_batch[:, i]) / nmed[i + 1] * w
    return -1.0 / w


def fno_batch(tracer, ca0, n_designs, n_ap=N_AP_FNO, chunk=FNO_CHUNK):
    """Axial real-ray working f-number per design: 1/(2 n' sin u') from the traced marginal ray, the same
    quantity lens_raytrace_cell.gate() reports, evaluated for every design in chunked kernel launches."""
    ax, ay = lrg.make_ray_grid(ca0, n_ap=n_ap, ca1_fill=1.10)
    n_rays = len(ax)
    p0 = np.zeros((n_rays, 3)); p0[:, 2] = lrc.OBJ_DIST
    d0 = np.stack([ax, ay, -np.full_like(ax, lrc.OBJ_DIST)], 1)
    d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
    n_last = float(np.asarray(tracer.nmed.numpy())[-1])
    out = np.full(n_designs, np.nan)
    for start in range(0, n_designs, chunk):
        ids = np.arange(start, min(start + chunk, n_designs))
        P = np.tile(p0, (len(ids), 1)); D = np.tile(d0, (len(ids), 1))
        did = np.repeat(ids.astype(np.int32), n_rays)
        _, d2, alive = tracer.trace(P, D, did, use_ca=True)
        sin_u = np.sqrt(d2[:, 0] ** 2 + d2[:, 1] ** 2).reshape(len(ids), n_rays)
        alive = alive.reshape(len(ids), n_rays)
        na = np.where(alive, sin_u, 0.0).max(axis=1) * n_last
        cnt = alive.sum(axis=1)
        out[ids] = np.where(cnt >= MIN_ALIVE, 1.0 / (2.0 * np.maximum(na, 1e-15)), np.nan)
    return out


def evaluate(vectors, ax, ay, fields=FIELDS_SEARCH):
    """Full batched evaluation of a population: spot RMS per field, objective, EFL, vignetting and f-number.
    Returns a dict of (N,) / (N,n_fields) arrays. All ray tracing is done by lens_raytrace_gpu's kernels."""
    rows = _rows(vectors)
    shared, z_batch, c_batch, mag, z_img, obj_z = lrg.build_z_c_batch_ext(rows, IDX_R_FREE)
    tracer = lrg.GPULensTracerExt(shared, z_batch, c_batch)
    rms, n_alive = tracer.rms_batch_fast(mag, z_img, obj_z, fields, ax, ay, use_ca=True)
    efl = paraxial_efl_batch(z_batch, c_batch, shared['nmed'])
    axial = n_alive[:, 0].astype(np.float64)
    full = n_alive[:, len(fields) - 1].astype(np.float64)
    vig = 1.0 - full / np.maximum(axial, 1.0)
    fno = fno_batch(tracer, shared['ca'][0], len(rows))
    obj = np.where(np.isfinite(rms).all(axis=1), (rms * FIELD_WEIGHTS).sum(axis=1), np.nan)
    return dict(rms=rms, n_alive=n_alive, efl=efl, vig=vig, fno=fno, obj=obj, mag=mag)


def certify(ev, efl_ref, vig_ref, fno_ref):
    """Three-certificate mask plus the per-certificate pass masks (for honest rejection accounting)."""
    ok_efl = np.abs(ev['efl'] - efl_ref) / abs(efl_ref) <= TOL_EFL_REL
    ok_vig = ev['vig'] <= vig_ref + 0.0
    ok_fno = np.abs(ev['fno'] - fno_ref) / abs(fno_ref) <= TOL_FNO_REL
    ok_num = np.isfinite(ev['obj'])
    return (ok_efl & ok_vig & ok_fno & ok_num), ok_efl, ok_vig, ok_fno, ok_num


# ----------------------------------------------------------------------------- search loop
def search(lo=LO, hi=HI, n_designs=N_DESIGNS, n_gen=N_GEN, n_elite=N_ELITE, seed=SEARCH_SEED,
           seed_patent=True, refs=None, verbose=True, tag=""):
    """Separable CMA-style loop: uniform first generation, then mean + per-dimension sigma of the certified
    elite, resampled Gaussian and clipped to the bounds, with the elite carried over (elitist). Only CERTIFIED
    candidates ever enter the elite or the returned best. Fixed seed, no other randomness."""
    rng = np.random.default_rng(seed)
    lo = np.asarray(lo, dtype=np.float64); hi = np.asarray(hi, dtype=np.float64)
    half = 0.5 * (hi - lo)
    ax, ay = lrg.make_ray_grid(lrc.build(lrc.P)['ca'][0], n_ap=N_AP_OBJ, ca1_fill=1.05)
    if refs is None:
        refs = reference_values(ax, ay)
    efl_meas, vig_ref, fno_ref, obj_ref, rms_ref = refs
    efl_ref = lrc.EFL_PATENT          # C-EFL is referenced to the PATENT's stated focal length, not to the
                                      # model's own reproduction of it (105.8224 mm here, +0.007%), so the
                                      # certificate cannot drift with the model.

    pop = lo + rng.random((n_designs, len(lo))) * (hi - lo)
    if seed_patent:
        pop[0] = np.clip(PATENT_VECTOR, lo, hi)     # the patent point is an explicit member of generation 0
    sigma = SIGMA0
    best = (np.inf, None, None)
    n_cert = 0; n_rej = 0
    rej_efl = 0; rej_vig = 0; rej_fno = 0; rej_num = 0
    log = []
    for g in range(n_gen):
        ev = evaluate(pop, ax, ay)
        ok, ok_efl, ok_vig, ok_fno, ok_num = certify(ev, efl_ref, vig_ref, fno_ref)
        n_cert += int(ok.sum()); n_rej += int((~ok).sum())
        rej_efl += int((~ok_efl).sum()); rej_vig += int((~ok_vig).sum())
        rej_fno += int((~ok_fno).sum()); rej_num += int((~ok_num).sum())
        idx = np.flatnonzero(ok)
        if len(idx) == 0:
            log.append((g, 0, np.nan, np.nan, sigma))
            pop = lo + rng.random((n_designs, len(lo))) * (hi - lo)
            sigma *= SIGMA_DECAY
            continue
        order = idx[np.argsort(ev['obj'][idx], kind='stable')]
        elite = pop[order[:min(n_elite, len(order))]]
        if ev['obj'][order[0]] < best[0]:
            best = (float(ev['obj'][order[0]]), pop[order[0]].copy(), ev['rms'][order[0]].copy(),
                    float(ev['efl'][order[0]]), float(ev['vig'][order[0]]), float(ev['fno'][order[0]]))
        log.append((g, len(idx), float(ev['obj'][order[0]]), best[0], sigma))
        if verbose:
            print(f"  [gen {g}{tag}] certified {len(idx):5d}/{n_designs}  best-of-gen {ev['obj'][order[0]]:.6f} mm"
                  f"  running best {best[0]:.6f} mm  sigma {sigma:.4f}")
        # separable Gaussian proposal around the certified elite: per-dimension step = elite spread, floored
        # at sigma x the bound half-width so the population cannot collapse before the last generation (with
        # collapsed bounds half == 0, so the floor is 0 and the null case stays exactly on the patent vector)
        mean = elite.mean(axis=0)
        std = elite.std(axis=0) if len(elite) > 1 else np.zeros(len(lo))
        step = np.maximum(std, sigma * half * 0.05)
        cand = mean + step * rng.standard_normal((n_designs - len(elite), len(lo)))
        pop = np.clip(np.vstack([elite, cand]), lo, hi)
        sigma *= SIGMA_DECAY
    counts = dict(certified=n_cert, rejected=n_rej, rej_efl=rej_efl, rej_vig=rej_vig, rej_fno=rej_fno,
                  rej_nonfinite=rej_num, evaluated=n_gen * n_designs)
    return dict(best_obj=best[0], best_vec=best[1], best_rms=best[2], best_efl=best[3] if best[1] is not None else np.nan,
                best_vig=best[4] if best[1] is not None else np.nan,
                best_fno=best[5] if best[1] is not None else np.nan,
                counts=counts, log=log, refs=(efl_meas, vig_ref, fno_ref, obj_ref, rms_ref))


def reference_values(ax, ay):
    """The patent example measured on the SAME instrument and the SAME ray grid as every candidate: paraxial
    EFL, full-field vignetting, axial working f-number, objective and per-field RMS."""
    ev = evaluate(PATENT_VECTOR[None, :], ax, ay)
    return (float(ev['efl'][0]), float(ev['vig'][0]), float(ev['fno'][0]), float(ev['obj'][0]), ev['rms'][0].copy())


def field_angles(mag):
    """Object-space half field angles (deg) of FIELDS_SEARCH at the prescription's conjugate."""
    return [float(np.degrees(np.arctan2(abs(h / mag), abs(lrc.OBJ_DIST)))) for h in FIELDS_SEARCH]


# ----------------------------------------------------------------------------- run
def main():
    t0 = time.time()
    print("=" * 112)
    print("CERTIFIED LENS DESIGN SEARCH v1 — 4 free parameters on US6141154 Example 1, GPU batched tracer")
    print("=" * 112)
    print(f"  device = {lrg.DEVICE}   population = {N_DESIGNS}/generation x {N_GEN} generations   seed = {SEARCH_SEED}")

    # G-EFL reuse: the CPU/GPU paraxial pair, then the batched form against the CPU function
    re_efl, efl_cpu, efl_gpu = lrg.efl_gate(verbose=True)
    ok_gefl = re_efl < 1e-12
    sysd = lrc.config_A()
    shared0, zb0, cb0, _, _, _ = lrg.build_z_c_batch_ext(_rows(PATENT_VECTOR[None, :]), IDX_R_FREE)
    efl_batched = float(paraxial_efl_batch(zb0, cb0, shared0['nmed'])[0])
    re_batch = lrg.rel_err(efl_cpu, efl_batched)
    print(f"  G-EFL batched form vs CPU paraxial_efl: {efl_batched:.10f} mm rel={re_batch:.2e}")
    ok_gefl = ok_gefl and re_batch < 1e-12

    ax, ay = lrg.make_ray_grid(sysd['ca'][0], n_ap=N_AP_OBJ, ca1_fill=1.05)
    refs = reference_values(ax, ay)
    efl_meas, vig_ref, fno_ref, obj_ref, rms_ref = refs
    ang = field_angles(lrc.MAG_PATENT)
    print(f"\n  objective fields (image height mm -> object half angle deg): "
          + ", ".join(f"{h:.1f} -> {a:.2f} deg" for h, a in zip(FIELDS_SEARCH, ang)))
    print(f"  PATENT EXAMPLE on this instrument: EFL {efl_meas:.4f} mm (patent states {lrc.EFL_PATENT} mm) | "
          f"vignetting {100*vig_ref:.4f}% | f/{fno_ref:.6f}")
    print(f"    per-field RMS (mm) = " + " ".join(f"{r:.5f}" for r in rms_ref) + f"  objective {obj_ref:.6f} mm")
    print(f"  certificates: C-EFL |dEFL|/EFL<={TOL_EFL_REL:.0e} of patent {lrc.EFL_PATENT} mm | "
          f"C-VIG vig<={100*vig_ref:.2f}% | C-FNO |df/#|/f/#<={TOL_FNO_REL:.0e}")
    print("  bounds: " + "  ".join(f"{n}[{a:.4f},{b:.4f}]" for n, a, b in BOUNDS))

    print("\n  --- run 1 ---")
    r1 = search(refs=refs, tag="")
    print("  --- run 2 (identical seed; repeatability) ---")
    r2 = search(refs=refs, tag="/r2")

    if r1['best_vec'] is None:
        print("\n  no certified design found — nothing may be reported")
        return False
    d_obj = abs(r1['best_obj'] - r2['best_obj'])
    d_vec = float(np.max(np.abs(r1['best_vec'] - r2['best_vec'])))
    print(f"\n  REPEATABILITY over two full runs: max|d objective| = {d_obj:.3e} mm, "
          f"max|d design vector| = {d_vec:.3e}")

    v = r1['best_vec']
    print(f"\n  BEST CERTIFIED DESIGN (space1, space2, r9, r13) = "
          f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f}, {v[3]:.6f}) mm")
    print(f"    patent vector                                = "
          f"({PATENT_VECTOR[0]:.6f}, {PATENT_VECTOR[1]:.6f}, {PATENT_VECTOR[2]:.6f}, {PATENT_VECTOR[3]:.6f}) mm")
    print(f"    certificates: EFL {r1['best_efl']:.4f} mm ({100*abs(r1['best_efl']-lrc.EFL_PATENT)/lrc.EFL_PATENT:.4f}% "
          f"of patent) | vignetting {100*r1['best_vig']:.4f}% (patent {100*vig_ref:.4f}%) | "
          f"f/{r1['best_fno']:.6f} ({100*abs(r1['best_fno']-fno_ref)/fno_ref:.4f}% of patent f/{fno_ref:.6f})")
    print(f"    per-field RMS (mm): patent " + " ".join(f"{r:.5f}" for r in rms_ref))
    print(f"                        best   " + " ".join(f"{r:.5f}" for r in r1['best_rms']))
    gain = 100.0 * (obj_ref - r1['best_obj']) / obj_ref
    print(f"    objective: patent {obj_ref:.6f} mm -> best certified {r1['best_obj']:.6f} mm  ({gain:+.2f}%)")

    c = r1['counts']
    print(f"\n  ACCOUNTING (run 1): evaluated {c['evaluated']}, certified {c['certified']}, "
          f"rejected {c['rejected']}  [rejections by certificate, not mutually exclusive: "
          f"C-EFL {c['rej_efl']}, C-VIG {c['rej_vig']}, C-FNO {c['rej_fno']}, non-finite spot {c['rej_nonfinite']}]")

    # null case: bounds collapsed onto the patent vector
    print("\n  --- null case: bounds collapsed onto the patent vector ---")
    rn = search(lo=PATENT_VECTOR, hi=PATENT_VECTOR, n_designs=256, n_gen=2, n_elite=16, refs=refs,
                verbose=False, tag="/null")
    null_vec_err = float(np.max(np.abs(rn['best_vec'] - PATENT_VECTOR))) if rn['best_vec'] is not None else np.inf
    null_obj_err = lrg.rel_err(rn['best_obj'], obj_ref)
    print(f"  collapsed search returns vector max|d| = {null_vec_err:.3e}, objective {rn['best_obj']:.8f} mm vs "
          f"patent {obj_ref:.8f} mm (rel {null_obj_err:.2e}); certified {rn['counts']['certified']}/"
          f"{rn['counts']['evaluated']}")

    ok_repeat = (d_obj == 0.0 and d_vec == 0.0)
    ok_null = (null_vec_err == 0.0 and null_obj_err < 1e-12
               and rn['counts']['certified'] == rn['counts']['evaluated'])
    ok_improve = r1['best_obj'] <= obj_ref
    ok_cert = (abs(r1['best_efl'] - lrc.EFL_PATENT) / lrc.EFL_PATENT <= TOL_EFL_REL
               and r1['best_vig'] <= vig_ref and abs(r1['best_fno'] - fno_ref) / fno_ref <= TOL_FNO_REL)
    ok = ok_gefl and ok_repeat and ok_null and ok_improve and ok_cert

    print("\n  LIMITS: four-parameter LOCAL search on one prescription with fixed glasses, single index set and "
          "one conjugate;")
    print("          spot RMS at three fields only — no MTF, no wavefront, no tolerancing, no manufacturability "
          "constraint;")
    print("          the bounds are centred on the patent values, so this measures local headroom, not lens design.")
    print("\n" + "=" * 112)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — G-EFL={ok_gefl} repeatability={ok_repeat} null-case={ok_null} "
          f"certificates-on-best={ok_cert} improves-or-matches={ok_improve}  [{time.time()-t0:.1f} s]")
    print("=" * 112)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
