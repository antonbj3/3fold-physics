"""Batched (N_designs x N_rays) port of the sequential-surface lens ray trace to warp/CUDA.

A CPU forward model evaluates one design per Python call (tens of milliseconds), which is the throughput ceiling
for anything that calls it repeatedly (an optimiser over lens spacings evaluates hundreds to thousands of designs
per run). Batching designs and rays into one kernel launch turns that into thousands of designs per second.

PORT, NOT REDESIGN: the same physics as the CPU tracer (sequential spherical/aspheric surfaces, exact
Newton-iteration surface intersection, vector Snell law, total-internal-reflection mask, aperture/stop mask); this
module must reproduce the CPU module closely, not approximate it. Two structural facts make it a clean port:
  (1) warp kernels are scalar per thread (one thread = one ray), unlike the numpy array-masked CPU code. A ray
      that goes bad (aspheric polynomial invalid outside its domain, total internal reflection, off aperture) sets
      alive=0 and breaks out of the surface loop, which is equivalent to the CPU's mask accumulation plus NaN
      propagation because dead rays are excluded from every downstream reduction on BOTH sides (see
      DEAD_RAY_EQUIVALENCE below).
  (2) the design family varies only surface THICKNESSES (z spacing) -- the two variable airspaces of the
      prescription -- never curvature, conic, polynomial, index or aperture. So one build() from the CPU module
      gives the SHARED per-surface constant tables once, and only the cumulative z array differs per design.
      build_z_batch() asserts that invariant, so a design-vector extension that breaks it fails loudly.
      The "ext" variant additionally frees individual radii and the object conjugate, which makes c per-design too.

DEAD_RAY_EQUIVALENCE (why the per-thread early break matches the CPU's vectorised NaN propagation):
  CPU: once a ray's Newton iterate goes non-finite (aspheric domain violated) or fails the total-internal-
  reflection / aperture test at surface i, the alive mask latches False at surface i; later surfaces still compute
  on that ray's poisoned state, but the result is never read because every consumer (RMS, 50%-encircled energy,
  vignetting counts) filters by the final alive mask first. GPU: the same latch plus an immediate break skips the
  wasted arithmetic instead of running it into garbage -- the output for dead rays is unspecified on both sides
  and masked out identically before every reduction. The Newton-loop early exit (per thread, step < 1e-10) is the
  same argument at finer grain.

PRECISION: float64 throughout (vec3d, float64 arrays). That is not the throughput-optimal choice (consumer GPUs
run float64 well below float32 rate) but it is what makes "matches the CPU within 1e-6 relative" achievable at
all: float32's ~7 significant digits sit at the noise floor of a 1e-6 gate applied to a Newton-iterated
intersection plus refraction chain over 20+ surfaces, so float32 was rejected up front rather than discovered to
fail afterwards. REPRODUCIBILITY: no cross-thread reductions (no atomics) appear in the ray-trace or paraxial
kernels -- every output element is written by exactly one thread from purely local state, so this kernel family is
exactly reproducible run to run by construction, unlike float-atomic reductions which are only statistically
reproducible. That is checked empirically anyway (N=5 repeats, repeatability_check) rather than asserted.

GATES (pre-registered, must pass before any throughput number is trusted):
  G-PORT: 10 random design vectors x the gate field heights -- GPU spot table against the CPU forward_spot_at --
          maximum relative deviation < 1e-6.
  G-EFL:  paraxial focal length of the patented prescription, GPU paraxial kernel against the CPU recursion.
  G-FNO:  the full focal-length / f-number / vignetting gate reproduced end to end on the GPU trace path.

I/O: no input files; prints the gate lines, a repeatability line and a throughput table. Requires a CUDA device
for the throughput numbers (it runs on warp's CPU backend, far below the batch sizes the benchmark assumes).
Prescription and CPU reference: lens_raytrace_cell in this package.
"""
import sys
import time
from pathlib import Path

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import lens_raytrace_cell as lrc

wp.init()
DEVICE = "cuda:0" if wp.get_cuda_device_count() > 0 else "cpu"

N_POLY = 6
GATE_FIELDS = [0.0, 50.0, 72.0]      # image-side field heights (mm) used by the gates and the benchmark


# --------------------------------------------------------------------------------------------- warp kernels (fp64)
@wp.func
def _sag_val(r2: wp.float64, c: wp.float64, k: wp.float64,
             p0: wp.float64, p1: wp.float64, p2: wp.float64, p3: wp.float64, p4: wp.float64, p5: wp.float64):
    a = wp.float64(1.0) - (wp.float64(1.0) + k) * c * c * r2
    bad = wp.int32(0)
    if a < wp.float64(0.0):
        a = wp.float64(0.0)
        bad = wp.int32(1)
    s = c * r2 / (wp.float64(1.0) + wp.sqrt(a))
    rp = r2 * r2
    s = s + p0 * rp; rp = rp * r2
    s = s + p1 * rp; rp = rp * r2
    s = s + p2 * rp; rp = rp * r2
    s = s + p3 * rp; rp = rp * r2
    s = s + p4 * rp; rp = rp * r2
    s = s + p5 * rp
    return s, bad


@wp.func
def _sag_dr_val(r: wp.float64, c: wp.float64, k: wp.float64,
                 p0: wp.float64, p1: wp.float64, p2: wp.float64, p3: wp.float64, p4: wp.float64, p5: wp.float64):
    r2 = r * r
    a = wp.float64(1.0) - (wp.float64(1.0) + k) * c * c * r2
    bad = wp.int32(0)
    if a < wp.float64(1.0e-14):
        bad = wp.int32(1)
        a = wp.float64(1.0e-14)   # guarded, matched by `bad` short-circuiting the caller (never trusted if bad=1)
    ds = c * r / wp.sqrt(a)
    rp = r2 * r
    ds = ds + wp.float64(4.0) * p0 * rp; rp = rp * r2
    ds = ds + wp.float64(6.0) * p1 * rp; rp = rp * r2
    ds = ds + wp.float64(8.0) * p2 * rp; rp = rp * r2
    ds = ds + wp.float64(10.0) * p3 * rp; rp = rp * r2
    ds = ds + wp.float64(12.0) * p4 * rp; rp = rp * r2
    ds = ds + wp.float64(14.0) * p5 * rp
    return ds, bad


@wp.kernel
def trace_kernel(
    design_id: wp.array(dtype=wp.int32),
    p_in: wp.array(dtype=wp.vec3d),
    d_in: wp.array(dtype=wp.vec3d),
    z: wp.array2d(dtype=wp.float64),          # (n_designs, n_surf)
    c_arr: wp.array(dtype=wp.float64),        # (n_surf,)
    k_arr: wp.array(dtype=wp.float64),
    poly: wp.array2d(dtype=wp.float64),       # (n_surf, 6)
    ca_arr: wp.array(dtype=wp.float64),
    poly_nz: wp.array(dtype=wp.int32),        # (n_surf,) 1 if any poly coeff nonzero
    nmed: wp.array(dtype=wp.float64),         # (n_surf+1,)
    n_surf: int,
    istop: int,
    use_ca: int,
    p_out: wp.array(dtype=wp.vec3d),
    d_out: wp.array(dtype=wp.vec3d),
    alive_out: wp.array(dtype=wp.int32),
):
    tid = wp.tid()
    did = design_id[tid]
    p = p_in[tid]
    d = d_in[tid]
    alive = wp.int32(1)
    for i in range(n_surf):
        if alive == 0:
            break
        zi = z[did, i]
        ci = c_arr[i]
        ki = k_arr[i]
        cai = ca_arr[i]
        n1 = nmed[i]
        n2 = nmed[i + 1]
        p0c = poly[i, 0]; p1c = poly[i, 1]; p2c = poly[i, 2]
        p3c = poly[i, 3]; p4c = poly[i, 4]; p5c = poly[i, 5]

        t = (zi - p[2]) / d[2]
        x = wp.float64(0.0)
        y = wp.float64(0.0)
        for _it in range(30):
            x = p[0] + t * d[0]
            y = p[1] + t * d[1]
            r2 = x * x + y * y
            sval, sbad = _sag_val(r2, ci, ki, p0c, p1c, p2c, p3c, p4c, p5c)
            if sbad == 1:
                alive = wp.int32(0)
                break
            f = (p[2] + t * d[2]) - (zi + sval)
            rr = wp.sqrt(wp.max(r2, wp.float64(1.0e-30)))
            dz, dbad = _sag_dr_val(rr, ci, ki, p0c, p1c, p2c, p3c, p4c, p5c)
            df = d[2] - dz * (x * d[0] + y * d[1]) / rr
            if dbad == 1 or wp.abs(df) < wp.float64(1.0e-12):
                alive = wp.int32(0)
                break
            step = f / df
            t = t - step
            if wp.abs(step) < wp.float64(1.0e-10):
                break
        if alive == 0:
            break

        # CPU recomputes the exit point AFTER the Newton loop as pn = p + t*d using the FULLY-UPDATED t (one more
        # implicit step than the x,y used for the last residual check inside the loop) -- must match that exactly,
        # not reuse the loop's last (pre-update-t) x,y, or near-aperture-edge rays can flip alive/dead (found via
        # the G-PORT gate: a 2-ray alive-count mismatch traced to exactly this off-by-one).
        x = p[0] + t * d[0]
        y = p[1] + t * d[1]
        r2f = x * x + y * y
        apply_ca = (use_ca == 1) or (i == istop)
        if apply_ca:
            if r2f > cai * cai * wp.float64(1.0 + 1.0e-9):
                alive = wp.int32(0)
                break
        else:
            if poly_nz[i] == 1:
                cap = wp.float64(1.2) * cai
                if r2f > cap * cap:
                    alive = wp.int32(0)
                    break

        rr = wp.sqrt(wp.max(r2f, wp.float64(1.0e-30)))
        dzr, dbad2 = _sag_dr_val(rr, ci, ki, p0c, p1c, p2c, p3c, p4c, p5c)
        if dbad2 == 1:
            alive = wp.int32(0)
            break
        Nx = -dzr * x / rr
        Ny = -dzr * y / rr
        Nz = wp.float64(1.0)
        nlen = wp.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
        Nx = Nx / nlen; Ny = Ny / nlen; Nz = Nz / nlen

        # exit point BEFORE refraction (uses the incoming direction d, exactly as CPU's pn = p + t*d does)
        pn = wp.vec3d(x, y, p[2] + t * d[2])

        cosi = -(d[0] * Nx + d[1] * Ny + d[2] * Nz)
        if cosi < wp.float64(0.0):
            Nx = -Nx; Ny = -Ny; Nz = -Nz
        cosi = wp.abs(cosi)
        eta = n1 / n2
        s2 = eta * eta * (wp.float64(1.0) - cosi * cosi)
        if s2 > wp.float64(1.0):
            alive = wp.int32(0)
            break
        cost = wp.sqrt(wp.max(wp.float64(1.0) - s2, wp.float64(0.0)))
        dx = eta * d[0] + (eta * cosi - cost) * Nx
        dy = eta * d[1] + (eta * cosi - cost) * Ny
        dzc = eta * d[2] + (eta * cosi - cost) * Nz
        dlen = wp.sqrt(dx * dx + dy * dy + dzc * dzc)
        d = wp.vec3d(dx / dlen, dy / dlen, dzc / dlen)
        p = pn

    p_out[tid] = p
    d_out[tid] = d
    alive_out[tid] = alive


@wp.kernel
def paraxial_efl_kernel(
    z: wp.array2d(dtype=wp.float64),      # (n_designs, n_surf)
    c_arr: wp.array(dtype=wp.float64),    # (n_surf,)
    nmed: wp.array(dtype=wp.float64),     # (n_surf+1,)
    n_surf: int,
    efl_out: wp.array(dtype=wp.float64),
):
    """One thread per design; literal port of lens_raytrace_cell.paraxial_efl's y/w recursion."""
    did = wp.tid()
    y = wp.float64(1.0)
    w = wp.float64(0.0)
    for i in range(n_surf):
        w = w - y * c_arr[i] * (nmed[i + 1] - nmed[i])
        if i < n_surf - 1:
            y = y + (z[did, i + 1] - z[did, i]) / nmed[i + 1] * w
    efl_out[did] = -wp.float64(1.0) / w


@wp.kernel
def gen_rays_kernel(
    ax: wp.array(dtype=wp.float64), ay: wp.array(dtype=wp.float64),   # (n_rays,) shared aperture grid
    H: wp.array(dtype=wp.float64),                                     # (n_groups,) screen height per group
    design_of_group: wp.array(dtype=wp.int32),                         # (n_groups,)
    obj_dist: wp.float64, n_rays: int,
    p_out: wp.array(dtype=wp.vec3d), d_out: wp.array(dtype=wp.vec3d), design_id_out: wp.array(dtype=wp.int32),
):
    """On-device port of forward_spot_at's ray generation (make_rays_from_screen with H=-h/|mag|) -- moves the
    numpy ray-construction (found to cost ~1s/call at 1024-design x 3-field batches, more than the trace kernel
    itself) onto the GPU so no host round-trip is needed before tracing."""
    tid = wp.tid()
    g = tid // n_rays
    r = tid - g * n_rays
    p0x = -H[g]
    dx = ax[r] - p0x
    dy = ay[r]
    dz = -obj_dist
    norm = wp.sqrt(dx * dx + dy * dy + dz * dz)
    p_out[tid] = wp.vec3d(p0x, wp.float64(0.0), obj_dist)
    d_out[tid] = wp.vec3d(dx / norm, dy / norm, dz / norm)
    design_id_out[tid] = design_of_group[g]


@wp.kernel
def reduce_rms_kernel(
    p: wp.array(dtype=wp.vec3d), d: wp.array(dtype=wp.vec3d), alive: wp.array(dtype=wp.int32),
    z_img_g: wp.array(dtype=wp.float64), n_rays: int,
    rms_out: wp.array(dtype=wp.float64), n_alive_out: wp.array(dtype=wp.int32),
):
    """One thread per (design,field) GROUP -- sequential two-pass reduction (mean, then variance) over that
    group's n_rays, entirely on-device, NO atomics (matches the project's float-atomics-are-only-statistically-
    reproducible caution: this kernel has none, and repeatability_check verifies that empirically). Computes
    RMS only (the quantity a spot-size objective consumes) -- the 50%-encircled-energy radius needs a per-row
    median/sort and is served by the numpy path in gpu_forward_spot_batch for gate/reporting use, not this
    throughput-path kernel."""
    g = wp.tid()
    zi = z_img_g[g]
    base = g * n_rays
    cnt = int(0)
    sx = wp.float64(0.0)
    sy = wp.float64(0.0)
    for r in range(n_rays):
        idx = base + r
        if alive[idx] == 1:
            pp = p[idx]; dd = d[idx]
            t = (zi - pp[2]) / dd[2]
            sx += pp[0] + t * dd[0]
            sy += pp[1] + t * dd[1]
            cnt += 1
    if cnt < 8:
        rms_out[g] = wp.float64(-1.0)   # NaN not used (device-side comparisons with nan are awkward); caller maps
        n_alive_out[g] = cnt            # cnt<8 -> NaN, matching forward_spot_at's MIN_ALIVE convention.
        return
    mx = sx / wp.float64(cnt)
    my = sy / wp.float64(cnt)
    ss = wp.float64(0.0)
    for r in range(n_rays):
        idx = base + r
        if alive[idx] == 1:
            pp = p[idx]; dd = d[idx]
            t = (zi - pp[2]) / dd[2]
            ex = pp[0] + t * dd[0] - mx
            ey = pp[1] + t * dd[1] - my
            ss += ex * ex + ey * ey
    rms_out[g] = wp.sqrt(ss / wp.float64(cnt))
    n_alive_out[g] = cnt


# --------------------------------------------------------------------------------------------- host-side batching
def _shared_tables(sysd0):
    """Extract the per-surface constant tables (c, k, poly, ca, nmed, poly_nz, istop, n_surf) from one build()
    dict -- SHARED across every design in a family (the design vector only moves z, never these)."""
    n_surf = len(sysd0['c'])
    poly_nz = (np.abs(sysd0['poly']) > 0).any(axis=1).astype(np.int32)
    return dict(n_surf=n_surf, c=sysd0['c'].astype(np.float64), k=sysd0['k'].astype(np.float64),
                poly=sysd0['poly'].astype(np.float64), ca=sysd0['ca'].astype(np.float64),
                nmed=sysd0['n'].astype(np.float64), poly_nz=poly_nz, istop=int(sysd0['istop']))


def build_z_batch(design_vectors, builder=lrc.config_spaces):
    """design_vectors: (N,2) array of (space1, space2), the prescription's two variable airspaces. Returns
    (shared_tables, z_batch (N,n_surf), mag (N,), z_img (N,)) -- calls the CPU module's own build()/paraxial
    functions per design (cheap, O(n_surf), not the bottleneck) and ASSERTS the shared tables are IDENTICAL
    across every design (the design vector must only move z; an extension that frees radii or indices would
    violate this and must fail loud)."""
    design_vectors = np.atleast_2d(np.asarray(design_vectors, dtype=np.float64))
    n = len(design_vectors)
    sysd0 = builder(space1=float(design_vectors[0, 0]), space2=float(design_vectors[0, 1]))
    shared = _shared_tables(sysd0)
    n_surf = shared['n_surf']
    z_batch = np.zeros((n, n_surf), dtype=np.float64)
    mag = np.zeros(n, dtype=np.float64)
    z_img = np.zeros(n, dtype=np.float64)
    for i, (s1, s2) in enumerate(design_vectors):
        sysd = builder(space1=float(s1), space2=float(s2))
        assert sysd['c'].shape == shared['c'].shape, "surface count changed across design vectors -- not a valid family"
        for key in ('c', 'k', 'poly', 'ca', 'n'):
            if not np.allclose(sysd[key], sysd0[key], rtol=0, atol=0):
                raise AssertionError(f"design vector changed shared table '{key}' -- the thicknesses-only "
                                      f"invariant is broken; the GPU batching in this module assumes shared "
                                      f"surface constants across the batch")
        z_batch[i] = sysd['z']
        zi, m = lrc.paraxial_mag_imgz(sysd, lrc.OBJ_DIST)
        z_img[i] = sysd['z_img']
        mag[i] = m
    return shared, z_batch, mag, z_img


class GPULensTracer:
    """Uploads one shared per-surface constant table + a batch of per-design z arrays once; repeated
    trace()/forward_spot_batch() calls reuse the device arrays (no re-upload of c/k/poly/ca per call)."""
    def __init__(self, shared, z_batch, device=DEVICE):
        self.device = device
        self.n_surf = shared['n_surf']
        self.istop = shared['istop']
        self.n_designs = z_batch.shape[0]
        self.z = wp.array(z_batch, dtype=wp.float64, device=device)
        self.c = wp.array(shared['c'], dtype=wp.float64, device=device)
        self.k = wp.array(shared['k'], dtype=wp.float64, device=device)
        self.poly = wp.array(shared['poly'], dtype=wp.float64, device=device)
        self.ca = wp.array(shared['ca'], dtype=wp.float64, device=device)
        self.poly_nz = wp.array(shared['poly_nz'], dtype=wp.int32, device=device)
        self.nmed = wp.array(shared['nmed'], dtype=wp.float64, device=device)

    def trace(self, p0, d0, design_id, use_ca=True):
        """p0,d0: (M,3) float64 numpy; design_id: (M,) int32 numpy in [0,n_designs). Returns (p,d,alive) numpy."""
        m = len(p0)
        p_in = wp.array(p0.astype(np.float64), dtype=wp.vec3d, device=self.device)
        d_in = wp.array(d0.astype(np.float64), dtype=wp.vec3d, device=self.device)
        did = wp.array(design_id.astype(np.int32), device=self.device)
        p_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        d_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        alive_out = wp.zeros(m, dtype=wp.int32, device=self.device)
        wp.launch(trace_kernel, dim=m,
                  inputs=[did, p_in, d_in, self.z, self.c, self.k, self.poly, self.ca, self.poly_nz, self.nmed,
                          self.n_surf, self.istop, int(1 if use_ca else 0)],
                  outputs=[p_out, d_out, alive_out], device=self.device)
        wp.synchronize()
        return p_out.numpy(), d_out.numpy(), alive_out.numpy().astype(bool)

    def paraxial_efl(self):
        out = wp.zeros(self.n_designs, dtype=wp.float64, device=self.device)
        wp.launch(paraxial_efl_kernel, dim=self.n_designs,
                  inputs=[self.z, self.c, self.nmed, self.n_surf], outputs=[out], device=self.device)
        wp.synchronize()
        return out.numpy()

    def rms_batch_fast(self, mag, z_img, fields, ax_h, ay_h, use_ca=True):
        """THROUGHPUT PATH: ray-gen, trace, and RMS reduction all stay on-device (only the tiny (n_groups,) RMS/
        n_alive arrays come back to host) -- this is what an optimiser loop
        actually needs per iteration. Returns (rms (n_designs,n_fields) with NaN where n_alive<8, n_alive same
        shape). ax_h,ay_h: (n_rays,) numpy aperture grid (shared across designs/fields)."""
        n_designs = self.n_designs
        n_fields = len(fields)
        n_groups = n_designs * n_fields
        n_rays = len(ax_h)

        H_h = np.zeros((n_designs, n_fields), dtype=np.float64)
        for fi, h in enumerate(fields):
            H_h[:, fi] = h / np.abs(mag) if h != 0.0 else 0.0
        design_of_group_h = np.repeat(np.arange(n_designs, dtype=np.int32), n_fields)
        z_img_g_h = np.repeat(np.asarray(z_img, dtype=np.float64), n_fields)

        ax_d = wp.array(ax_h.astype(np.float64), dtype=wp.float64, device=self.device)
        ay_d = wp.array(ay_h.astype(np.float64), dtype=wp.float64, device=self.device)
        H_d = wp.array(H_h.reshape(-1), dtype=wp.float64, device=self.device)
        dog_d = wp.array(design_of_group_h, dtype=wp.int32, device=self.device)
        z_img_g_d = wp.array(z_img_g_h, dtype=wp.float64, device=self.device)

        m = n_groups * n_rays
        p_in = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        d_in = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        did = wp.zeros(m, dtype=wp.int32, device=self.device)
        wp.launch(gen_rays_kernel, dim=m,
                  inputs=[ax_d, ay_d, H_d, dog_d, lrc.OBJ_DIST, n_rays],
                  outputs=[p_in, d_in, did], device=self.device)

        p_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        d_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        alive_out = wp.zeros(m, dtype=wp.int32, device=self.device)
        wp.launch(trace_kernel, dim=m,
                  inputs=[did, p_in, d_in, self.z, self.c, self.k, self.poly, self.ca, self.poly_nz, self.nmed,
                          self.n_surf, self.istop, int(1 if use_ca else 0)],
                  outputs=[p_out, d_out, alive_out], device=self.device)

        rms_out = wp.zeros(n_groups, dtype=wp.float64, device=self.device)
        n_alive_out = wp.zeros(n_groups, dtype=wp.int32, device=self.device)
        wp.launch(reduce_rms_kernel, dim=n_groups,
                  inputs=[p_out, d_out, alive_out, z_img_g_d, n_rays],
                  outputs=[rms_out, n_alive_out], device=self.device)
        wp.synchronize()

        rms = rms_out.numpy().reshape(n_designs, n_fields)
        n_alive = n_alive_out.numpy().reshape(n_designs, n_fields)
        rms = np.where(n_alive < 8, np.nan, rms)
        return rms, n_alive


# --------------------------------------------------------------------------------------------- EXTENDED FAMILY
# A population-based optimiser wants a wider design vector on the GPU (the two variable airspaces, the object
# conjugate and a few free radii), not just the thicknesses-only family above. Two of those dimensions violate
# build_z_batch's shared-table assumption:
#   (1) radii{surf_idx: R} -> changes c[i]=1/R PER DESIGN (population members differ in curvature, not just z)
#   (2) the conjugate -> obj_z = -abs(conjugate_m)*1000 PER DESIGN (object distance varies, so ray generation's
#       object-space z AND the per-design mag/z_img from paraxial_mag_imgz(sysd, obj_z) both vary per design)
# k, poly, ca, poly_nz, nmed stay SHARED (radii dict only overrides Q[i][0]=R, never k/poly/n/ca; see
# lens_raytrace_cell.config_spaces) -- only c and z become per-design arrays, exactly like z already was.
@wp.kernel
def trace_kernel_ext(
    design_id: wp.array(dtype=wp.int32),
    p_in: wp.array(dtype=wp.vec3d),
    d_in: wp.array(dtype=wp.vec3d),
    z: wp.array2d(dtype=wp.float64),          # (n_designs, n_surf)
    c: wp.array2d(dtype=wp.float64),          # (n_designs, n_surf) -- PER-DESIGN now (free radii)
    k_arr: wp.array(dtype=wp.float64),
    poly: wp.array2d(dtype=wp.float64),
    ca_arr: wp.array(dtype=wp.float64),
    poly_nz: wp.array(dtype=wp.int32),
    nmed: wp.array(dtype=wp.float64),
    n_surf: int,
    istop: int,
    use_ca: int,
    p_out: wp.array(dtype=wp.vec3d),
    d_out: wp.array(dtype=wp.vec3d),
    alive_out: wp.array(dtype=wp.int32),
):
    """Literal copy of trace_kernel's body -- the ONLY change is ci = c[did, i] (per-design) instead of
    c_arr[i] (shared). Kept as a separate kernel (not a branch inside trace_kernel) so the thicknesses-only
    hot path pays zero extra indexing cost."""
    tid = wp.tid()
    did = design_id[tid]
    p = p_in[tid]
    d = d_in[tid]
    alive = wp.int32(1)
    for i in range(n_surf):
        if alive == 0:
            break
        zi = z[did, i]
        ci = c[did, i]
        ki = k_arr[i]
        cai = ca_arr[i]
        n1 = nmed[i]
        n2 = nmed[i + 1]
        p0c = poly[i, 0]; p1c = poly[i, 1]; p2c = poly[i, 2]
        p3c = poly[i, 3]; p4c = poly[i, 4]; p5c = poly[i, 5]

        t = (zi - p[2]) / d[2]
        x = wp.float64(0.0)
        y = wp.float64(0.0)
        for _it in range(30):
            x = p[0] + t * d[0]
            y = p[1] + t * d[1]
            r2 = x * x + y * y
            sval, sbad = _sag_val(r2, ci, ki, p0c, p1c, p2c, p3c, p4c, p5c)
            if sbad == 1:
                alive = wp.int32(0)
                break
            f = (p[2] + t * d[2]) - (zi + sval)
            rr = wp.sqrt(wp.max(r2, wp.float64(1.0e-30)))
            dz, dbad = _sag_dr_val(rr, ci, ki, p0c, p1c, p2c, p3c, p4c, p5c)
            df = d[2] - dz * (x * d[0] + y * d[1]) / rr
            if dbad == 1 or wp.abs(df) < wp.float64(1.0e-12):
                alive = wp.int32(0)
                break
            step = f / df
            t = t - step
            if wp.abs(step) < wp.float64(1.0e-10):
                break
        if alive == 0:
            break

        x = p[0] + t * d[0]
        y = p[1] + t * d[1]
        r2f = x * x + y * y
        apply_ca = (use_ca == 1) or (i == istop)
        if apply_ca:
            if r2f > cai * cai * wp.float64(1.0 + 1.0e-9):
                alive = wp.int32(0)
                break
        else:
            if poly_nz[i] == 1:
                cap = wp.float64(1.2) * cai
                if r2f > cap * cap:
                    alive = wp.int32(0)
                    break

        rr = wp.sqrt(wp.max(r2f, wp.float64(1.0e-30)))
        dzr, dbad2 = _sag_dr_val(rr, ci, ki, p0c, p1c, p2c, p3c, p4c, p5c)
        if dbad2 == 1:
            alive = wp.int32(0)
            break
        Nx = -dzr * x / rr
        Ny = -dzr * y / rr
        Nz = wp.float64(1.0)
        nlen = wp.sqrt(Nx * Nx + Ny * Ny + Nz * Nz)
        Nx = Nx / nlen; Ny = Ny / nlen; Nz = Nz / nlen

        pn = wp.vec3d(x, y, p[2] + t * d[2])

        cosi = -(d[0] * Nx + d[1] * Ny + d[2] * Nz)
        if cosi < wp.float64(0.0):
            Nx = -Nx; Ny = -Ny; Nz = -Nz
        cosi = wp.abs(cosi)
        eta = n1 / n2
        s2 = eta * eta * (wp.float64(1.0) - cosi * cosi)
        if s2 > wp.float64(1.0):
            alive = wp.int32(0)
            break
        cost = wp.sqrt(wp.max(wp.float64(1.0) - s2, wp.float64(0.0)))
        dx = eta * d[0] + (eta * cosi - cost) * Nx
        dy = eta * d[1] + (eta * cosi - cost) * Ny
        dzc = eta * d[2] + (eta * cosi - cost) * Nz
        dlen = wp.sqrt(dx * dx + dy * dy + dzc * dzc)
        d = wp.vec3d(dx / dlen, dy / dlen, dzc / dlen)
        p = pn

    p_out[tid] = p
    d_out[tid] = d
    alive_out[tid] = alive


@wp.kernel
def gen_rays_kernel_ext(
    ax: wp.array(dtype=wp.float64), ay: wp.array(dtype=wp.float64),
    H: wp.array(dtype=wp.float64),
    obj_z_g: wp.array(dtype=wp.float64),      # (n_groups,) PER-GROUP object-plane z (the conjugate varies per design)
    design_of_group: wp.array(dtype=wp.int32),
    n_rays: int,
    p_out: wp.array(dtype=wp.vec3d), d_out: wp.array(dtype=wp.vec3d), design_id_out: wp.array(dtype=wp.int32),
):
    """Same as gen_rays_kernel but obj_dist is per-group (obj_z_g), not one shared scalar -- required once
    the object conjugate is a free design dimension (each population member can sit at a different conjugate)."""
    tid = wp.tid()
    g = tid // n_rays
    r = tid - g * n_rays
    oz = obj_z_g[g]
    p0x = -H[g]
    dx = ax[r] - p0x
    dy = ay[r]
    dz = -oz
    norm = wp.sqrt(dx * dx + dy * dy + dz * dz)
    p_out[tid] = wp.vec3d(p0x, wp.float64(0.0), oz)
    d_out[tid] = wp.vec3d(dx / norm, dy / norm, dz / norm)
    design_id_out[tid] = design_of_group[g]


def build_z_c_batch_ext(design_vectors, free_idx, builder=lrc.config_spaces):
    """Extended version of build_z_batch: design_vectors (N,3+len(free_idx)) rows = (space1, space2,
    conjugate_m, r_i for i in free_idx). Returns (shared_tables [k,poly,ca,nmed,poly_nz,istop,n_surf], z_batch,
    c_batch (both (N,n_surf) -- PER-DESIGN), mag (N,), z_img (N,), obj_z (N,) -- obj_z = -abs(conjugate_m)*1000,
    i.e. the object distance in millimetres). Asserts k/poly/ca/nmed are IDENTICAL across the batch (freeing a
    radius only moves c) -- the same fail-loud discipline as build_z_batch's assertion."""
    design_vectors = np.atleast_2d(np.asarray(design_vectors, dtype=np.float64))
    n = len(design_vectors)
    n_r = len(free_idx)

    def _build_one(row):
        s1, s2, conj_m = row[0], row[1], row[2]
        radii = {int(i): float(v) for i, v in zip(free_idx, row[3:3 + n_r])}
        return builder(space1=float(s1), space2=float(s2), radii=radii), float(conj_m)

    sysd0, _ = _build_one(design_vectors[0])
    n_surf = len(sysd0['c'])
    shared = dict(n_surf=n_surf, k=sysd0['k'].astype(np.float64), poly=sysd0['poly'].astype(np.float64),
                  ca=sysd0['ca'].astype(np.float64), nmed=sysd0['n'].astype(np.float64),
                  poly_nz=(np.abs(sysd0['poly']) > 0).any(axis=1).astype(np.int32), istop=int(sysd0['istop']))
    z_batch = np.zeros((n, n_surf), dtype=np.float64)
    c_batch = np.zeros((n, n_surf), dtype=np.float64)
    mag = np.zeros(n, dtype=np.float64)
    z_img = np.zeros(n, dtype=np.float64)
    obj_z = np.zeros(n, dtype=np.float64)
    for i, row in enumerate(design_vectors):
        sysd, conj_m = _build_one(row)
        assert sysd['c'].shape == (n_surf,), "surface count changed across design vectors -- not a valid family"
        for key, shkey in (('k', 'k'), ('poly', 'poly'), ('ca', 'ca'), ('n', 'nmed')):
            if not np.allclose(sysd[key], shared[shkey], rtol=0, atol=0):
                raise AssertionError(f"design vector changed shared table '{key}' -- the radii contract is "
                                      f"broken (a freed radius must only move c=1/R, never {key})")
        z_batch[i] = sysd['z']
        c_batch[i] = sysd['c']
        oz = -abs(conj_m) * 1000.0
        _, m = lrc.paraxial_mag_imgz(sysd, oz)
        z_img[i] = sysd['z_img']
        mag[i] = m
        obj_z[i] = oz
    return shared, z_batch, c_batch, mag, z_img, obj_z


class GPULensTracerExt:
    """Extended (free conjugate + free radii) counterpart of GPULensTracer -- c and z are BOTH per-design device
    arrays (uploaded once per generation); k/poly/ca/nmed stay shared (uploaded once, reused)."""
    def __init__(self, shared, z_batch, c_batch, device=DEVICE):
        self.device = device
        self.n_surf = shared['n_surf']
        self.istop = shared['istop']
        self.n_designs = z_batch.shape[0]
        self.z = wp.array(z_batch, dtype=wp.float64, device=device)
        self.c = wp.array(c_batch, dtype=wp.float64, device=device)
        self.k = wp.array(shared['k'], dtype=wp.float64, device=device)
        self.poly = wp.array(shared['poly'], dtype=wp.float64, device=device)
        self.ca = wp.array(shared['ca'], dtype=wp.float64, device=device)
        self.poly_nz = wp.array(shared['poly_nz'], dtype=wp.int32, device=device)
        self.nmed = wp.array(shared['nmed'], dtype=wp.float64, device=device)

    def trace(self, p0, d0, design_id, use_ca=True):
        m = len(p0)
        p_in = wp.array(p0.astype(np.float64), dtype=wp.vec3d, device=self.device)
        d_in = wp.array(d0.astype(np.float64), dtype=wp.vec3d, device=self.device)
        did = wp.array(design_id.astype(np.int32), device=self.device)
        p_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        d_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        alive_out = wp.zeros(m, dtype=wp.int32, device=self.device)
        wp.launch(trace_kernel_ext, dim=m,
                  inputs=[did, p_in, d_in, self.z, self.c, self.k, self.poly, self.ca, self.poly_nz, self.nmed,
                          self.n_surf, self.istop, int(1 if use_ca else 0)],
                  outputs=[p_out, d_out, alive_out], device=self.device)
        wp.synchronize()
        return p_out.numpy(), d_out.numpy(), alive_out.numpy().astype(bool)

    def rms_batch_fast(self, mag, z_img, obj_z, fields, ax_h, ay_h, use_ca=True):
        """Device-fused throughput path (ray-gen + trace + RMS reduction all on-device) -- the per-generation
        call of a population optimiser. obj_z: (n_designs,) per-design object-plane z."""
        n_designs = self.n_designs
        n_fields = len(fields)
        n_groups = n_designs * n_fields
        n_rays = len(ax_h)

        H_h = np.zeros((n_designs, n_fields), dtype=np.float64)
        for fi, h in enumerate(fields):
            H_h[:, fi] = h / np.abs(mag) if h != 0.0 else 0.0
        design_of_group_h = np.repeat(np.arange(n_designs, dtype=np.int32), n_fields)
        z_img_g_h = np.repeat(np.asarray(z_img, dtype=np.float64), n_fields)
        obj_z_g_h = np.repeat(np.asarray(obj_z, dtype=np.float64), n_fields)

        ax_d = wp.array(ax_h.astype(np.float64), dtype=wp.float64, device=self.device)
        ay_d = wp.array(ay_h.astype(np.float64), dtype=wp.float64, device=self.device)
        H_d = wp.array(H_h.reshape(-1), dtype=wp.float64, device=self.device)
        dog_d = wp.array(design_of_group_h, dtype=wp.int32, device=self.device)
        z_img_g_d = wp.array(z_img_g_h, dtype=wp.float64, device=self.device)
        obj_z_g_d = wp.array(obj_z_g_h, dtype=wp.float64, device=self.device)

        m = n_groups * n_rays
        p_in = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        d_in = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        did = wp.zeros(m, dtype=wp.int32, device=self.device)
        wp.launch(gen_rays_kernel_ext, dim=m,
                  inputs=[ax_d, ay_d, H_d, obj_z_g_d, dog_d, n_rays],
                  outputs=[p_in, d_in, did], device=self.device)

        p_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        d_out = wp.zeros(m, dtype=wp.vec3d, device=self.device)
        alive_out = wp.zeros(m, dtype=wp.int32, device=self.device)
        wp.launch(trace_kernel_ext, dim=m,
                  inputs=[did, p_in, d_in, self.z, self.c, self.k, self.poly, self.ca, self.poly_nz, self.nmed,
                          self.n_surf, self.istop, int(1 if use_ca else 0)],
                  outputs=[p_out, d_out, alive_out], device=self.device)

        rms_out = wp.zeros(n_groups, dtype=wp.float64, device=self.device)
        n_alive_out = wp.zeros(n_groups, dtype=wp.int32, device=self.device)
        wp.launch(reduce_rms_kernel, dim=n_groups,
                  inputs=[p_out, d_out, alive_out, z_img_g_d, n_rays],
                  outputs=[rms_out, n_alive_out], device=self.device)
        wp.synchronize()

        rms = rms_out.numpy().reshape(n_designs, n_fields)
        n_alive = n_alive_out.numpy().reshape(n_designs, n_fields)
        rms = np.where(n_alive < 8, np.nan, rms)
        return rms, n_alive


def port_gate_ext(n_designs=5, n_ap=81, free_idx=(8, 9, 13),
                   lo=(25.0, 1.0, 1.5, -239.4, 214.8, -497.6),
                   hi=(40.0, 8.0, 8.0, -143.6, 358.0, -298.6), verbose=True):
    """G-PORT-EXT: n_designs random extended (space1, space2, conjugate_m, r8, r9, r13) vectors x GATE_FIELDS --
    GPU (rms_batch_fast, the throughput call path) vs CPU lrc.forward_spot_at at matching obj_z/mag/n_ap.
    Returns (max_rel_err, rows).
    NOTE: n_ap default (81) is a literal, not N_AP_GATE -- this function is defined ABOVE N_AP_GATE/RNG_SEED
    in the file (before the original INSTRUMENT GATE section); default args are bound at def-time so it can't
    reference those later globals, only look them up inside the body (RNG_SEED below IS a body-time lookup)."""
    rng = np.random.default_rng(RNG_SEED + 3)
    lo = np.asarray(lo); hi = np.asarray(hi)
    dv = lo + rng.random((n_designs, len(lo))) * (hi - lo)
    shared, z_batch, c_batch, mag, z_img, obj_z = build_z_c_batch_ext(dv, free_idx)
    tracer = GPULensTracerExt(shared, z_batch, c_batch)
    ax, ay = make_ray_grid(shared['ca'][0], n_ap=n_ap, ca1_fill=1.05)
    rms_g, n_alive_g = tracer.rms_batch_fast(mag, z_img, obj_z, GATE_FIELDS, ax, ay, use_ca=True)

    rows = []
    max_rel = 0.0
    for di in range(n_designs):
        s1, s2, conj_m = dv[di, 0], dv[di, 1], dv[di, 2]
        radii = {int(i): float(v) for i, v in zip(free_idx, dv[di, 3:3 + len(free_idx)])}
        sysd = lrc.config_spaces(space1=float(s1), space2=float(s2), radii=radii)
        oz = -abs(float(conj_m)) * 1000.0
        _, m_cpu = lrc.paraxial_mag_imgz(sysd, oz)
        for fi, h in enumerate(GATE_FIELDS):
            rms_c, ee_c, n_c = lrc.forward_spot_at(sysd, h, m_cpu, oz, use_ca=True, n_ap=n_ap)
            re = rel_err(rms_c, rms_g[di, fi])
            max_rel = max(max_rel, re)
            rows.append((di, h, n_c, int(n_alive_g[di, fi]), rms_c, rms_g[di, fi], re))
            if verbose:
                flag = "OK" if (re < 1e-6 and n_c == n_alive_g[di, fi]) else "MISMATCH"
                print(f"  [ext] design={di} h={h:6.1f} n_alive cpu/gpu={n_c}/{int(n_alive_g[di,fi])} "
                      f"RMS cpu={rms_c:.10f} gpu={rms_g[di,fi]:.10f} rel={re:.2e}  [{flag}]")
    return max_rel, rows


def make_ray_grid(ca0, n_ap=81, ca1_fill=1.05):
    """Shared aperture sampling grid (design-independent: front group / ca[0] is unmodified by the design vector)."""
    r1 = ca0 * ca1_fill
    a = np.linspace(-r1, r1, n_ap)
    ax, ay = np.meshgrid(a, a)
    mm = ax ** 2 + ay ** 2 <= r1 ** 2
    return ax[mm], ay[mm]


def _vectorized_rms_ee50(xy, alive, z_img, p, d):
    """Fully vectorized (no per-design Python loop) port of rms_radius/ee50_radius over a (n_groups, n_rays) batch.
    xy: (n_groups, n_rays, 2) image-plane points (already NaN where the ray is dead-or-below-min-alive is handled
    by the caller); alive: (n_groups, n_rays) bool. Returns (rms, ee50, n_alive) arrays, shape (n_groups,).
    This replaced an earlier per-design `for di in range(n_designs): ...` reduction that was the ACTUAL throughput
    bottleneck (the GPU trace kernel itself sustains ~15M rays/s; the Python-level reduction loop did not) --
    measured via isolated timing of the trace() call vs the full batch call before rewriting this."""
    n_g, n_r = alive.shape
    n_alive = alive.sum(1)
    safe_n = np.maximum(n_alive, 1)
    x = np.where(alive, xy[:, :, 0], 0.0)
    y = np.where(alive, xy[:, :, 1], 0.0)
    mx = x.sum(1) / safe_n
    my = y.sum(1) / safe_n
    dx = np.where(alive, x - mx[:, None], 0.0)
    dy = np.where(alive, y - my[:, None], 0.0)
    var = (dx ** 2 + dy ** 2).sum(1) / safe_n
    rms = np.sqrt(var)
    dist2 = np.where(alive, dx ** 2 + dy ** 2, np.inf)
    order = np.argsort(dist2, axis=1)
    dist_sorted = np.sqrt(np.take_along_axis(dist2, order, axis=1))
    lo = np.clip((n_alive - 1) // 2, 0, n_r - 1)
    hi = np.clip(n_alive // 2, 0, n_r - 1)
    rows = np.arange(n_g)
    ee50 = 0.5 * (dist_sorted[rows, lo] + dist_sorted[rows, hi])
    bad = n_alive < 8
    rms = np.where(bad, np.nan, rms)
    ee50 = np.where(bad, np.nan, ee50)
    return rms, ee50, n_alive


def gpu_forward_spot_batch(tracer, mag, z_img, fields, ax, ay, use_ca=True):
    """Batched port of lens_raytrace_cell.forward_spot_at over ALL designs x ALL fields x ALL rays in ONE trace
    kernel launch (fields fused into the design axis: group index = design_idx*n_fields + field_idx, all sharing
    the same underlying design's z-table via design_id). Returns dict (design_idx, field) -> (rms, ee50, n_alive).
    """
    n_designs = tracer.n_designs
    n_rays = len(ax)
    n_fields = len(fields)
    n_groups = n_designs * n_fields

    H = np.zeros((n_designs, n_fields))
    for fi, h in enumerate(fields):
        H[:, fi] = h / np.abs(mag) if h != 0.0 else 0.0
    p0 = np.zeros((n_groups, n_rays, 3), dtype=np.float64)
    p0[:, :, 0] = -H.reshape(-1, 1)
    p0[:, :, 2] = lrc.OBJ_DIST
    d0 = np.zeros((n_groups, n_rays, 3), dtype=np.float64)
    d0[:, :, 0] = ax[None, :] - p0[:, :, 0]
    d0[:, :, 1] = ay[None, :]
    d0[:, :, 2] = -lrc.OBJ_DIST
    d0 /= np.linalg.norm(d0, axis=2, keepdims=True)
    design_id = np.repeat(np.arange(n_designs, dtype=np.int32), n_fields * n_rays)
    p0f = p0.reshape(-1, 3); d0f = d0.reshape(-1, 3)

    p2, d2, alive = tracer.trace(p0f, d0f, design_id, use_ca=use_ca)
    alive2 = alive.reshape(n_groups, n_rays)
    p2r = p2.reshape(n_groups, n_rays, 3)
    d2r = d2.reshape(n_groups, n_rays, 3)

    z_img_g = np.repeat(z_img, n_fields)   # (n_groups,) -- z_img per design, broadcast over fields
    denom = np.where(alive2, d2r[:, :, 2], 1.0)
    t = (z_img_g[:, None] - p2r[:, :, 2]) / denom
    xp = p2r[:, :, 0] + t * d2r[:, :, 0]
    yp = p2r[:, :, 1] + t * d2r[:, :, 1]
    xy = np.stack([xp, yp], axis=2)

    rms, ee50, n_alive = _vectorized_rms_ee50(xy, alive2, z_img_g, p2r, d2r)
    results = {}
    for di in range(n_designs):
        for fi, h in enumerate(fields):
            g = di * n_fields + fi
            results[(di, h)] = (float(rms[g]), float(ee50[g]), int(n_alive[g]))
    return results


# --------------------------------------------------------------------------------------------- INSTRUMENT GATE
RNG_SEED = 20260716
N_AP_GATE = 81


def rel_err(a, b):
    a, b = float(a), float(b)
    if not (np.isfinite(a) and np.isfinite(b)):
        return np.inf if np.isfinite(a) != np.isfinite(b) else 0.0
    denom = max(abs(a), abs(b), 1e-12)
    return abs(a - b) / denom


def port_gate(n_designs=10, n_ap=N_AP_GATE, verbose=True):
    """G-PORT: n_designs random (space1, space2) vectors x GATE_FIELDS -- GPU spot table vs CPU
    lens_raytrace_cell.forward_spot_at, same rays (n_ap), same physics. Returns (max_rel_err, rows)."""
    rng = np.random.default_rng(RNG_SEED)
    lo = np.array([20.0, 1.0]); hi = np.array([40.0, 8.0])
    dv = lo + rng.random((n_designs, 2)) * (hi - lo)
    shared, z_batch, mag, z_img = build_z_batch(dv)
    tracer = GPULensTracer(shared, z_batch)
    ax, ay = make_ray_grid(shared['ca'][0], n_ap=n_ap, ca1_fill=1.05)
    gpu_res = gpu_forward_spot_batch(tracer, mag, z_img, GATE_FIELDS, ax, ay, use_ca=True)

    rows = []
    max_rel = 0.0
    for di in range(n_designs):
        sysd = lrc.config_spaces(space1=float(dv[di, 0]), space2=float(dv[di, 1]))
        _, m_cpu = lrc.paraxial_mag_imgz(sysd, lrc.OBJ_DIST)
        for h in GATE_FIELDS:
            rms_c, ee_c, n_c = lrc.forward_spot_at(sysd, h, m_cpu, lrc.OBJ_DIST, use_ca=True, n_ap=n_ap)
            rms_g, ee_g, n_g = gpu_res[(di, h)]
            re_rms = rel_err(rms_c, rms_g)
            re_ee = rel_err(ee_c, ee_g)
            max_rel = max(max_rel, re_rms, re_ee)
            rows.append((di, h, n_c, n_g, rms_c, rms_g, re_rms, ee_c, ee_g, re_ee))
            if verbose:
                flag = "OK" if (re_rms < 1e-6 and re_ee < 1e-6 and n_c == n_g) else "MISMATCH"
                print(f"  design={di:2d} h={h:6.1f} n_alive cpu/gpu={n_c:5d}/{n_g:5d} "
                      f"RMS cpu={rms_c:.10f} gpu={rms_g:.10f} rel={re_rms:.2e}  "
                      f"EE50 cpu={ee_c:.10f} gpu={ee_g:.10f} rel={re_ee:.2e}  [{flag}]")
    return max_rel, rows


def efl_gate(verbose=True):
    """G-EFL: config_A paraxial EFL, GPU paraxial kernel vs lens_raytrace_cell.paraxial_efl (literal port)."""
    sysd = lrc.config_A()
    efl_cpu = lrc.paraxial_efl(sysd)
    shared = _shared_tables(sysd)
    tracer = GPULensTracer(shared, sysd['z'][None, :])
    efl_gpu = float(tracer.paraxial_efl()[0])
    re = rel_err(efl_cpu, efl_gpu)
    if verbose:
        print(f"  G-EFL: CPU={efl_cpu:.10f}mm  GPU={efl_gpu:.10f}mm  rel={re:.2e}  patent=105.815mm")
    return re, efl_cpu, efl_gpu


def fno_vignetting_gate(verbose=True):
    """G-FNO: reproduce lens_raytrace_cell.gate()'s working f/# + vignetting on the GPU trace path, config_A,
    single design (N=1) -- axial marginal-ray NA at the image + full-field/axial alive-ray ratio."""
    sysd = lrc.config_A()
    shared = _shared_tables(sysd)
    tracer = GPULensTracer(shared, sysd['z'][None, :])

    def rays_h(h, n_ap, ca1_fill=1.10):
        # literal port of lens_raytrace_cell.make_rays_from_screen(sysd, H=h, ...) -- NO negation (gate() calls
        # make_rays_from_screen directly with raw field values, unlike forward_spot's -H convention).
        ax, ay = make_ray_grid(sysd['ca'][0], n_ap=n_ap, ca1_fill=ca1_fill)
        p0 = np.zeros((len(ax), 3)); p0[:, 0] = h; p0[:, 2] = lrc.OBJ_DIST
        d0 = np.stack([ax - p0[:, 0], ay, -np.full_like(ax, lrc.OBJ_DIST)], 1)
        d0 /= np.linalg.norm(d0, axis=1, keepdims=True)
        return p0, d0

    p0, d0 = rays_h(0.0, 201, 1.10)
    did = np.zeros(len(p0), dtype=np.int32)
    p2, d2, alive = tracer.trace(p0, d0, did, use_ca=True)
    NA_gpu = sysd['n'][-1] * np.sqrt(d2[alive, 0] ** 2 + d2[alive, 1] ** 2).max()
    fno_gpu = 1.0 / (2.0 * NA_gpu)

    # NOTE: the CPU gate()'s vignetting rays use the DEFAULT ca1_fill=1.05 (only the NA call above overrides to 1.10) --
    # must match that exactly (an earlier version of this gate reused 1.10 here and got a spurious ~2.4% vignetting
    # mismatch purely from a different aperture-fill grid, not a physics bug -- caught by re-deriving gate()'s
    # actual call signature rather than assuming a shared default).
    pF, dF = rays_h(2664.0, 121, 1.05)
    _, _, aF = tracer.trace(pF, dF, np.zeros(len(pF), dtype=np.int32), use_ca=True)
    pA, dA = rays_h(0.0, 121, 1.05)
    _, _, aA = tracer.trace(pA, dA, np.zeros(len(pA), dtype=np.int32), use_ca=True)
    vig_gpu = 1.0 - aF.sum() / max(aA.sum(), 1)

    efl_c, fno_c, m_c, zi_c, vig_c, ok_efl_c, ok_fno_c = lrc.gate()
    re_fno = rel_err(fno_c, fno_gpu)
    re_vig = rel_err(vig_c, vig_gpu)
    if verbose:
        print(f"  G-FNO: CPU f/{fno_c:.4f}  GPU f/{fno_gpu:.4f}  rel={re_fno:.2e}  (patent f/0.99)")
        print(f"  vignetting@full-field: CPU={vig_c:.4f}  GPU={vig_gpu:.4f}  rel={re_vig:.2e}  (patent Table 4 ~29%)")
    return re_fno, re_vig, fno_gpu, vig_gpu


def repeatability_check(n_designs=8, n_ap=51, n_repeats=5):
    """HONEST NOTE per task: measure run-to-run GPU variance, don't assume determinism. No atomic reductions
    appear in trace_kernel/reduce_rms_kernel/paraxial_efl_kernel (every output written once by its own thread),
    so this SHOULD be exactly bit-reproducible -- verified empirically (both the numpy-reduction path and the
    device-fused rms_batch_fast path) rather than just argued from the no-atomics structural claim."""
    rng = np.random.default_rng(RNG_SEED + 1)
    lo = np.array([20.0, 1.0]); hi = np.array([40.0, 8.0])
    dv = lo + rng.random((n_designs, 2)) * (hi - lo)
    shared, z_batch, mag, z_img = build_z_batch(dv)
    tracer = GPULensTracer(shared, z_batch)
    ax, ay = make_ray_grid(shared['ca'][0], n_ap=n_ap, ca1_fill=1.05)
    runs, runs_fast = [], []
    for _ in range(n_repeats):
        res = gpu_forward_spot_batch(tracer, mag, z_img, GATE_FIELDS, ax, ay, use_ca=True)
        runs.append(np.array([res[(di, h)][0] for di in range(n_designs) for h in GATE_FIELDS]))
        rms_f, _ = tracer.rms_batch_fast(mag, z_img, GATE_FIELDS, ax, ay, use_ca=True)
        runs_fast.append(rms_f.reshape(-1))
    runs = np.stack(runs); runs_fast = np.stack(runs_fast)
    spread = runs.max(0) - runs.min(0)
    spread_fast = runs_fast.max(0) - runs_fast.min(0)
    return float(spread.max()), float(np.nanmean(runs)), float(np.nanmax(spread_fast))


# --------------------------------------------------------------------------------------------- BENCHMARK
def fast_rms_gate(n_designs=10, n_ap=N_AP_GATE, verbose=True):
    """G-PORT-FAST: same as port_gate but through rms_batch_fast (on-device ray-gen + reduction, the path an
    optimizer loop actually calls) -- checked SEPARATELY from port_gate's numpy-reduction path since they are
    different code (device kernel vs host numpy), not just different entry points to the same math."""
    rng = np.random.default_rng(RNG_SEED)
    lo = np.array([20.0, 1.0]); hi = np.array([40.0, 8.0])
    dv = lo + rng.random((n_designs, 2)) * (hi - lo)
    shared, z_batch, mag, z_img = build_z_batch(dv)
    tracer = GPULensTracer(shared, z_batch)
    ax, ay = make_ray_grid(shared['ca'][0], n_ap=n_ap, ca1_fill=1.05)
    rms_g, n_alive_g = tracer.rms_batch_fast(mag, z_img, GATE_FIELDS, ax, ay, use_ca=True)

    max_rel = 0.0
    for di in range(n_designs):
        sysd = lrc.config_spaces(space1=float(dv[di, 0]), space2=float(dv[di, 1]))
        _, m_cpu = lrc.paraxial_mag_imgz(sysd, lrc.OBJ_DIST)
        for fi, h in enumerate(GATE_FIELDS):
            rms_c, _, n_c = lrc.forward_spot_at(sysd, h, m_cpu, lrc.OBJ_DIST, use_ca=True, n_ap=n_ap)
            re = rel_err(rms_c, rms_g[di, fi])
            max_rel = max(max_rel, re)
            if verbose and (re >= 1e-6 or n_c != n_alive_g[di, fi]):
                print(f"  [fast] design={di} h={h} n_alive cpu/gpu={n_c}/{n_alive_g[di,fi]} "
                      f"RMS cpu={rms_c:.10f} gpu={rms_g[di,fi]:.10f} rel={re:.2e}")
    if verbose:
        print(f"  G-PORT-FAST (device-fused RMS path) max rel error = {max_rel:.2e}")
    return max_rel


def benchmark(batch_sizes=(32, 256, 1024), n_ap=N_AP_GATE, fields=GATE_FIELDS):
    rng = np.random.default_rng(RNG_SEED + 2)
    lo = np.array([20.0, 1.0]); hi = np.array([40.0, 8.0])
    rows = []
    for nb in batch_sizes:
        dv = lo + rng.random((nb, 2)) * (hi - lo)
        shared, z_batch, mag, z_img = build_z_batch(dv)
        tracer = GPULensTracer(shared, z_batch)
        ax, ay = make_ray_grid(shared['ca'][0], n_ap=n_ap, ca1_fill=1.05)
        n_rays = len(ax)

        # warm-up (module load / kernel compile is a one-time cost, excluded from the timed region)
        _ = gpu_forward_spot_batch(tracer, mag, z_img, fields, ax[:50], ay[:50], use_ca=True)
        t0 = time.perf_counter()
        _ = gpu_forward_spot_batch(tracer, mag, z_img, fields, ax, ay, use_ca=True)
        t_gpu_full = time.perf_counter() - t0
        evals_s_gpu_full = nb / t_gpu_full

        # fast path: on-device ray-gen + RMS reduction (host round-trip is (n_designs,n_fields) floats only) --
        # the throughput number that matters for a spot-size objective (RMS-only consumer)
        _ = tracer.rms_batch_fast(mag, z_img, fields, ax[:50], ay[:50], use_ca=True)
        t0 = time.perf_counter()
        _ = tracer.rms_batch_fast(mag, z_img, fields, ax, ay, use_ca=True)
        t_gpu_fast = time.perf_counter() - t0
        evals_s_gpu_fast = nb / t_gpu_fast

        # CPU baseline: same design vectors, same fields, same n_ap, via lens_raytrace_cell directly
        n_cpu = min(nb, 64)   # cap CPU timing sample (CPU is ~30ms/design -- 1024 designs would take minutes)
        t0 = time.perf_counter()
        for i in range(n_cpu):
            sysd = lrc.config_spaces(space1=float(dv[i, 0]), space2=float(dv[i, 1]))
            _, m = lrc.paraxial_mag_imgz(sysd, lrc.OBJ_DIST)
            for h in fields:
                lrc.forward_spot_at(sysd, h, m, lrc.OBJ_DIST, use_ca=True, n_ap=n_ap)
        t_cpu = time.perf_counter() - t0
        evals_s_cpu = n_cpu / t_cpu

        rows.append((nb, n_rays, evals_s_gpu_full, evals_s_gpu_fast, evals_s_cpu,
                      evals_s_gpu_full / evals_s_cpu, evals_s_gpu_fast / evals_s_cpu))
    return rows


def main():
    print("=" * 100)
    print("LENS RAYTRACE GPU — instrument gate + benchmark")
    print(f"device={DEVICE}  warp={wp.config.version if hasattr(wp, 'config') else '?'}")
    print("=" * 100)

    print("\n[G-EFL] paraxial EFL, config_A, GPU vs CPU")
    re_efl, efl_c, efl_g = efl_gate()

    print("\n[G-FNO] patent f/# + vignetting gate, config_A, GPU trace path vs CPU gate()")
    re_fno, re_vig, fno_g, vig_g = fno_vignetting_gate()

    print("\n[G-PORT] 10 random design vectors x GATE_FIELDS, GPU spot table vs CPU forward_spot_at")
    max_rel_port, rows = port_gate(n_designs=10, n_ap=N_AP_GATE)

    print("\n[G-PORT-FAST] same 10 design vectors, through the device-fused rms_batch_fast throughput path")
    max_rel_fast = fast_rms_gate(n_designs=10, n_ap=N_AP_GATE)

    print("\n[REPEATABILITY] N=5 repeats, 8 designs x 51^2-grid rays -- run-to-run RMS spread")
    spread, mean_rms, spread_fast = repeatability_check()
    print(f"  numpy-reduction path max spread = {spread:.3e} mm  (mean RMS ~{mean_rms:.4f}mm)")
    print(f"  device-fused fast path max spread = {spread_fast:.3e} mm  (no atomics in either kernel path)")

    print("\n[BENCHMARK] evals/second, GPU-batched vs CPU sequential, batch sizes 32/256/1024")
    rows_b = benchmark()
    print("  batch | rays/design | GPU full evals/s | GPU fast evals/s | CPU evals/s (sampled) | speedup(full) | speedup(fast)")
    for nb, nr, gsf, gsfast, cs, spf, spfast in rows_b:
        print(f"  {nb:5d} | {nr:11d} | {gsf:16.1f} | {gsfast:16.1f} | {cs:21.1f} | {spf:12.1f}x | {spfast:12.1f}x")

    print("\n" + "=" * 100)
    print("GATE SUMMARY")
    print(f"  G-EFL  rel error = {re_efl:.2e}  ({'PASS' if re_efl < 1e-6 else 'FAIL'} @1e-6)")
    print(f"  G-FNO  rel error = {re_fno:.2e}  ({'PASS' if re_fno < 1e-6 else 'FAIL'} @1e-6)")
    print(f"  G-VIG  rel error = {re_vig:.2e}")
    print(f"  G-PORT      max rel error (10 designs x 3 fields, RMS+EE50, numpy path) = {max_rel_port:.2e}  "
          f"({'PASS' if max_rel_port < 1e-6 else 'FAIL'} @1e-6)")
    print(f"  G-PORT-FAST max rel error (same 10 designs, device-fused RMS path)      = {max_rel_fast:.2e}  "
          f"({'PASS' if max_rel_fast < 1e-6 else 'FAIL'} @1e-6)")
    print(f"  repeatability: numpy-path max spread = {spread:.3e} mm, fast-path max spread = {spread_fast:.3e} mm")
    print("=" * 100)


if __name__ == "__main__":
    main()

