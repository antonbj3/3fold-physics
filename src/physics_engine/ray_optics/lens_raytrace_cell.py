"""Sequential-surface ray trace for a rotationally symmetric lens, on a public patent prescription.

PRESCRIPTION: US 6,141,154 (Kreitzer) Example 1 / Table 1 — f/0.99, effective focal length 105.815 mm, 10 elements
in 3 units, two variable airspaces (the patent's "double focus" adjustment). The image-side stack (cover plate
BAC1 7 mm, index-matched fluid 420550 4.3 mm, faceplate 539570 7 mm) is flat; the image surface radius is 3657 mm
(near flat). Transcribed from the patent full text. Indices: the published table's index column did not survive
text extraction, so the catalogue e-line values of the NAMED glasses are used (FC5 1.48914, BACD18 1.64128, BACD5
1.59142, FD6 1.81265, BSC7 1.51872, BAC1 1.57487, acrylic 1.4935) together with the index-coded fluids
(420550 -> 1.420, 539570 -> 1.539). With those indices the paraxial focal length lands at 105.822 mm against the
patent's 105.815 mm (+0.007%), which jointly confirms radii, thicknesses and indices.

GEOMETRY (no heuristics, numpy only):
  - aspheric sag z(r) = c r^2/(1+sqrt(1-(1+k) c^2 r^2)) + D r^4 + E r^6 + F r^8 + G r^10 + H r^12 + I r^14, c=1/R
  - exact ray-surface intersection by Newton iteration; vector Snell law at the exact surface normal; total
    internal reflection masked out
  - aspheric polynomials are meaningless outside the design aperture, so rays beyond 1.2x the clear aperture of a
    polynomial surface are killed even in the aberration-only (no clear-aperture) mode; the stop is ALWAYS enforced
  - rays are traced in the design direction (object plane -> image surface) and spots are measured ON the image
    surface; the object-side spot is the image spot x |1/m|, cross-checked by an explicit reverse trace at the
    field heights where the reverse launch cone is valid

GATES: G-EFL the paraxial focal length reproduces the patent's 105.815 mm within 1%; G-FNO the real-ray working
f/# at the image (1/(2 n' sin u'), axial) reproduces f/0.99 within 2%; external anchor: the patent's Table 4
"vignetting losses 29% at full field" against the traced full-field/axial throughput ratio.

I/O: no input files; run directly to print the gate lines, the spot table over field heights and a verdict
(exit 0 on pass). Spot radii are RMS and 50%-encircled-energy, in millimetres at the image surface.
"""
import numpy as np

# ----------------------------------------------------------------------------- prescription
# US6141154 Example 1 (surfaces numbered from the object side; thickness = to next surface; CA = diameter)
INF = 0.0  # flat marker (curvature 0)
P = [
    # R,          t,        n_after, k,        D..I (y^4..y^14),                                                   CAd,    tag
    ( 2901.1860, 15.00,     1.4935,  0.0,      [ 1.5357e-07,-1.1470e-11, 5.4733e-16, 1.8080e-20,-2.4692e-24, 7.0415e-29], 238.00, "S1  L1 acrylic"),
    (  209.1157, 60.46,     1.0,     4.0,      [ 1.0916e-07,-9.1896e-12,-4.4716e-16, 7.3390e-20, 4.1308e-24,-6.1243e-28], 184.25, "S2"),
    ( -296.4704, 13.00,     1.48914, 0.0,      None,                                                                174.26, "S3  FC5"),
    (  296.4704, 10.00,     1.0,     0.0,      None,                                                                161.54, "S4"),
    (  392.6675, 16.00,     1.4935, -25.0,     [-5.5549e-08,-2.0289e-11, 1.1519e-15, 2.4383e-19,-1.9504e-23, 1.3099e-27], 161.29, "S5  L3 acrylic"),
    (  540.9802, 67.72,     1.0,     0.0,      [ 6.9500e-08,-1.1849e-11, 1.1142e-15, 1.0660e-19,-3.1801e-24,-3.9466e-28], 161.88, "S6"),
    (  INF,       1.91,     1.0,     0.0,      None,                                                                209.12, "S7  STOP"),
    ( 3148.5892, 41.00,     1.64128, 0.0,      None,                                                                213.21, "S8  BACD18"),
    ( -189.1711, 29.300,    1.0,     0.0,      None,                                                                216.25, "S9  [Space1]"),
    (  287.1786, 42.50,     1.64128, 0.0,      None,                                                                226.61, "S10 BACD18"),
    ( -473.3888,  0.39,     1.0,     0.0,      None,                                                                225.40, "S11"),
    (  233.4301, 44.00,     1.59142, 0.0,      None,                                                                206.27, "S12 BACD5"),
    ( -400.1245,  0.05,     1.0,     0.0,      None,                                                                201.84, "S13"),
    ( -398.7020,  9.00,     1.81265, 0.0,      None,                                                                201.83, "S14 FD6"),
    (  135.8890, 53.87,     1.51872, 0.0,      None,                                                                178.59, "S15 BSC7"),
    ( -287.2871, 11.20,     1.0,     0.0,      None,                                                                177.52, "S16"),
    ( -408.5138, 17.50,     1.4935,  0.0,      [-8.0884e-08,-1.2768e-11,-5.5910e-16,-1.8983e-20, 4.4793e-24, 5.0401e-28], 174.84, "S17 L9 acrylic"),
    ( -233.0149, 64.47,     1.0,    -6.1068,   [-5.2904e-08,-1.2418e-11,-9.2265e-16, 6.0394e-20, 5.7518e-24,-9.2069e-29], 178.30, "S18"),
    ( -119.5886,  6.79,     1.4935,  0.0,      [ 7.6271e-08, 4.4580e-11,-3.4805e-14, 9.6328e-18,-1.1505e-21, 5.2675e-26], 152.00, "S19 field flattener"),
    (-29999.9997, 2.520,    1.0,     0.0,      None,                                                                157.00, "S20 [Space2]"),
    (  INF,       7.00,     1.57487, 0.0,      None,                                                                161.00, "S21 cover BAC1"),
    (  INF,       4.30,     1.420,   0.0,      None,                                                                161.00, "S22 fluid 420550"),
    (  INF,       7.00,     1.539,   0.0,      None,                                                                161.00, "S23 faceplate 539570"),
]
R_PHOSPHOR = 3657.0           # image surface radius magnitude (Example 1); the sign is resolved by the trace
OBJ_DIST   = -3876.02         # object plane, from the S1 vertex (focus position 1)
EFL_PATENT, FNO_PATENT = 105.815, 0.99
MAG_PATENT = -0.0270
IDX_SPACE1, IDX_SPACE2 = 8, 19          # the two variable airspaces of Example 1 (S9 and S20 thicknesses)
FIELDS = [0.0, 50.0, 72.0, 98.0, 140.0, 198.0]   # image-side field heights, mm (72.0 = the design image semi-height)

def build(P):
    n_surf = len(P)
    z = np.zeros(n_surf); c = np.zeros(n_surf); k = np.zeros(n_surf)
    poly = np.zeros((n_surf, 6)); n_med = np.ones(n_surf + 1); ca = np.zeros(n_surf)
    zc = 0.0
    for i, (R, t, n_after, kk, co, cad, tag) in enumerate(P):
        z[i] = zc; zc += t
        c[i] = 0.0 if R == 0.0 else 1.0 / R
        k[i] = kk
        if co is not None: poly[i] = co
        n_med[i + 1] = n_after
        ca[i] = cad / 2.0
    istop = next((i for i, row in enumerate(P) if "STOP" in row[6]), -1)   # -1 = no stop tag (e.g. a
    # sub-prescription sliced out for reflection tracing, section I) -- harmless: istop is only consulted
    # via `i == sysd['istop']` in trace(), which never matches a real surface index when -1
    return dict(z=z, c=c, k=k, poly=poly, n=n_med, ca=ca, z_img=zc, istop=istop)

def sag(r2, c, k, poly):
    a = 1.0 - (1.0 + k) * c * c * r2
    bad = a < 0.0
    a = np.where(bad, 0.0, a)
    s = c * r2 / (1.0 + np.sqrt(a))
    rp = r2 * r2
    for j in range(6):
        s = s + poly[j] * rp
        rp = rp * r2
    return np.where(bad, np.nan, s)

def sag_dr(r, c, k, poly):
    r2 = r * r
    a = 1.0 - (1.0 + k) * c * c * r2
    a = np.where(a < 1e-14, np.nan, a)
    ds = c * r / np.sqrt(a)
    rp = r2 * r
    for j in range(6):
        ds = ds + (4 + 2 * j) * poly[j] * rp
        rp = rp * r2
    return ds

def trace(sysd, p, d, reverse=False, use_ca=True, record=None):
    """Trace rays through all surfaces (reverse=True walks image -> object). Returns p, d, alive.

    record: OPTIONAL list, additive and numerically inert -- if a list is passed,
    (surface_index, p_at_surface.copy(), alive.copy()) is appended for EVERY surface. Needed by consumers that
    require the ray's actual refracted polyline inside the lens rather than a straight chord between the entry
    and exit planes (the chord leaves the glass and creates spurious obstructions)."""
    z, c, k, poly, nmed, ca = sysd['z'], sysd['c'], sysd['k'], sysd['poly'], sysd['n'], sysd['ca']
    order = range(len(z) - 1, -1, -1) if reverse else range(len(z))
    p = p.copy(); d = d.copy()
    alive = np.ones(len(p), bool)
    np.seterr(all='ignore')
    for i in order:
        n1 = nmed[i + 1] if reverse else nmed[i]
        n2 = nmed[i] if reverse else nmed[i + 1]
        t = (z[i] - p[:, 2]) / d[:, 2]
        for _ in range(30):
            x = p[:, 0] + t * d[:, 0]; y = p[:, 1] + t * d[:, 1]
            r2 = x * x + y * y
            f = (p[:, 2] + t * d[:, 2]) - (z[i] + sag(r2, c[i], k[i], poly[i]))
            r = np.sqrt(np.maximum(r2, 1e-30))
            dz = sag_dr(r, c[i], k[i], poly[i])
            df = d[:, 2] - dz * (x * d[:, 0] + y * d[:, 1]) / r
            step = f / np.where(np.abs(df) < 1e-12, np.nan, df)
            t = t - step
            st = step[alive & np.isfinite(step)] if alive.any() else np.array([])
            if st.size == 0 or np.max(np.abs(st)) < 1e-10: break
        pn = p + t[:, None] * d
        r2 = pn[:, 0] ** 2 + pn[:, 1] ** 2
        alive &= np.isfinite(t) & np.isfinite(r2)
        if use_ca or i == sysd['istop']:
            alive &= r2 <= ca[i] ** 2 * (1 + 1e-9)
        elif np.any(poly[i]):   # asphere polynomial invalid far outside design aperture
            alive &= r2 <= (1.2 * ca[i]) ** 2
        if record is not None:
            record.append((int(i), pn.copy(), alive.copy()))
        r = np.sqrt(np.maximum(r2, 1e-30))
        dzr = sag_dr(r, c[i], k[i], poly[i])
        N = np.stack([-dzr * pn[:, 0] / r, -dzr * pn[:, 1] / r, np.ones_like(r)], 1)
        N /= np.linalg.norm(N, axis=1, keepdims=True)
        cosi = -np.einsum('ij,ij->i', d, N)
        N = np.where(cosi[:, None] < 0, -N, N)
        cosi = np.abs(cosi)
        eta = n1 / n2
        s2 = eta * eta * (1.0 - cosi * cosi)
        alive &= s2 <= 1.0
        cost = np.sqrt(np.clip(1.0 - s2, 0.0, 1.0))
        d = eta * d + (eta * cosi - cost)[:, None] * N
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        p = pn
        alive &= np.isfinite(p).all(1) & np.isfinite(d).all(1)
    return p, d, alive

def to_plane(p, d, zp):
    t = (zp - p[:, 2]) / d[:, 2]
    return p + t[:, None] * d

def to_curved(p, d, z0, sag_fn, iters=4):
    """Propagate to surface z = z0 + sag_fn(r); fixed-point iteration (near-flat surfaces)."""
    zt = np.full(len(p), z0)
    for _ in range(iters):
        t = (zt - p[:, 2]) / d[:, 2]
        pp = p + t[:, None] * d
        zt = z0 + sag_fn(np.sqrt(pp[:, 0] ** 2 + pp[:, 1] ** 2))
    t = (zt - p[:, 2]) / d[:, 2]
    return p + t[:, None] * d

def paraxial_efl(sysd):
    z, c, nmed = sysd['z'], sysd['c'], sysd['n']
    y, w = 1.0, 0.0
    for i in range(len(z)):
        w = w - y * c[i] * (nmed[i + 1] - nmed[i])
        if i < len(z) - 1:
            y = y + (z[i + 1] - z[i]) / nmed[i + 1] * w
    return -1.0 / w

def paraxial_mag_imgz(sysd, obj_z):
    z, c, nmed = sysd['z'], sysd['c'], sysd['n']
    y = 1.0; w = nmed[0] * y / (z[0] - obj_z)
    w0 = w
    for i in range(len(z)):
        if i > 0: y = y + (z[i] - z[i - 1]) / nmed[i] * w
        w = w - y * c[i] * (nmed[i + 1] - nmed[i])
    t_img = -y / w * nmed[-1]
    return z[-1] + t_img, w0 / w

def make_rays_from_screen(sysd, H, n_ap=81, ca1_fill=1.05):
    r1 = sysd['ca'][0] * ca1_fill
    a = np.linspace(-r1, r1, n_ap)
    ax, ay = np.meshgrid(a, a)
    mm = ax ** 2 + ay ** 2 <= r1 ** 2
    ax, ay = ax[mm], ay[mm]
    p0 = np.array([H, 0.0, OBJ_DIST])
    d = np.stack([ax, ay, np.zeros_like(ax)], 1) - p0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.tile(p0, (len(d), 1)), d

def rms_radius(xy):
    cxy = xy - xy.mean(0)
    return float(np.sqrt((cxy ** 2).sum(1).mean()))

def ee50_radius(xy):
    cxy = xy - xy.mean(0)
    return float(np.median(np.sqrt((cxy ** 2).sum(1))))

def forward_spot(sysd, h, mag, use_ca=True, img_sag=None, zoff=0.0, n_ap=81):
    """Trace object -> image from the object point conjugate to image height h; spot at the image surface.
    Returns (rms, ee50, n_rays, centroid_h) in image-side mm."""
    H = h / abs(mag) if h else 0.0
    p, d = make_rays_from_screen(sysd, -H, n_ap=n_ap)   # negative side: real image inverts
    p2, d2, alive = trace(sysd, p, d, reverse=False, use_ca=use_ca)
    if alive.sum() < 8: return np.nan, np.nan, int(alive.sum()), np.nan
    p2, d2 = p2[alive], d2[alive]
    if img_sag is None:
        pp = to_plane(p2, d2, sysd['z_img'] + zoff)
    else:
        pp = to_curved(p2, d2, sysd['z_img'] + zoff, img_sag)
    return rms_radius(pp[:, :2]), ee50_radius(pp[:, :2]), int(alive.sum()), float(abs(pp[:, 0].mean()))

def best_focus(sysd, h, mag, use_ca=True, zscan=np.linspace(-25.0, 10.0, 351), n_ap=81):
    H = h / abs(mag) if h else 0.0
    p, d = make_rays_from_screen(sysd, -H, n_ap=n_ap)
    p2, d2, alive = trace(sysd, p, d, reverse=False, use_ca=use_ca)
    if alive.sum() < 8: return np.nan, np.nan, int(alive.sum())
    p2, d2 = p2[alive], d2[alive]
    best = (np.inf, 0.0)
    for dz in zscan:
        r = rms_radius(to_plane(p2, d2, sysd['z_img'] + dz)[:, :2])
        if r < best[0]: best = (r, dz)
    return best[1], best[0], int(alive.sum())

def reverse_check(sysd, h, n_ap=81, fill=1.35):
    """Reverse trace image -> object with a stop-aimed cone (valid at low fields): object-side RMS."""
    ist = sysd['istop']
    rs, zs = sysd['ca'][ist] * fill, sysd['z'][ist]
    a = np.linspace(-rs, rs, n_ap)
    ax, ay = np.meshgrid(a, a)
    mm = ax ** 2 + ay ** 2 <= rs ** 2
    ax, ay = ax[mm], ay[mm]
    p0 = np.array([h, 0.0, sysd['z_img']])
    d = np.stack([ax, ay, np.full_like(ax, zs)], 1) - p0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    p2, d2, alive = trace(sysd, np.tile(p0, (len(d), 1)), d, reverse=True, use_ca=True)
    if alive.sum() < 8: return np.nan, 0
    ps = to_plane(p2[alive], d2[alive], OBJ_DIST)
    return rms_radius(ps[:, :2]), int(alive.sum())

def config_A():
    return build(P)
def config_spaces(space1=None, space2=None, radii=None):
    """Example 1 with its two variable airspaces (and optionally individual radii) overridden. Only thicknesses
    (and, if given, curvatures) change: conics, polynomials, indices and clear apertures stay as patented."""
    Q = [list(s) for s in P]
    if space1 is not None: Q[IDX_SPACE1][1] = float(space1)
    if space2 is not None: Q[IDX_SPACE2][1] = float(space2)
    if radii:
        for i, Rn in radii.items():
            Q[int(i)][0] = float(Rn)
    return build([tuple(q) for q in Q])


def gate():
    sysd = config_A()
    efl = paraxial_efl(sysd)
    zi, m = paraxial_mag_imgz(sysd, OBJ_DIST)
    p, d = make_rays_from_screen(sysd, 0.0, n_ap=201, ca1_fill=1.10)
    p2, d2, alive = trace(sysd, p, d, reverse=False, use_ca=True)
    NA = sysd['n'][-1] * np.sqrt(d2[alive, 0] ** 2 + d2[alive, 1] ** 2).max()
    fno = 1.0 / (2.0 * NA)
    pF, dF = make_rays_from_screen(sysd, 2664.0, n_ap=121)
    _, _, aF = trace(sysd, pF, dF, reverse=False, use_ca=True)
    pA, dA = make_rays_from_screen(sysd, 0.0, n_ap=121)
    _, _, aA = trace(sysd, pA, dA, reverse=False, use_ca=True)
    vig_rel_ax = 1.0 - aF.sum() / max(aA.sum(), 1)
    ok_efl = abs(efl - EFL_PATENT) / EFL_PATENT < 0.01
    ok_fno = abs(fno - FNO_PATENT) / FNO_PATENT < 0.02
    return efl, fno, m, zi, vig_rel_ax, ok_efl, ok_fno

def spot_table_fwd(sysd, mag, img_sag=None, zoff=0.0):
    rows = []
    for h in FIELDS:
        rms1, ee1, n1, _ = forward_spot(sysd, h, mag, use_ca=True, img_sag=img_sag, zoff=zoff)
        rms0, ee0, n0, _ = forward_spot(sysd, h, mag, use_ca=False, img_sag=img_sag, zoff=zoff)
        rows.append((h, n1, rms1, ee1, n0, rms0, ee0))
    return rows

def print_tab(rows, mag):
    print("  h_img   | rays(CA)  RMS(CA)   EE50(CA) | rays(noCA) RMS(noCA) EE50(noCA) |  RMS@object(CA)  [all mm]")
    for h, n1, r1, e1, n0, r0, e0 in rows:
        scr = r1 / abs(mag) if np.isfinite(r1) else np.nan
        print(f"  {h:7.1f} | {n1:8d} {r1:9.4f} {e1:9.4f} | {n0:9d} {r0:9.4f} {e0:9.4f} | {scr:12.3f}")

def make_rays_from_screen_at(sysd, H, obj_z, n_ap=81, ca1_fill=1.05):
    r1 = sysd['ca'][0] * ca1_fill
    a = np.linspace(-r1, r1, n_ap)
    ax, ay = np.meshgrid(a, a)
    mm = ax ** 2 + ay ** 2 <= r1 ** 2
    ax, ay = ax[mm], ay[mm]
    p0 = np.array([H, 0.0, obj_z])
    d = np.stack([ax, ay, np.zeros_like(ax)], 1) - p0
    d /= np.linalg.norm(d, axis=1, keepdims=True)
    return np.tile(p0, (len(d), 1)), d

def forward_spot_at(sysd, h, mag, obj_z, use_ca=True, zoff=0.0, n_ap=81):
    H = h / abs(mag) if h else 0.0
    p, d = make_rays_from_screen_at(sysd, -H, obj_z, n_ap=n_ap)
    p2, d2, alive = trace(sysd, p, d, reverse=False, use_ca=use_ca)
    if alive.sum() < 8: return np.nan, np.nan, int(alive.sum())
    p2, d2 = p2[alive], d2[alive]
    pp = to_plane(p2, d2, sysd['z_img'] + zoff)
    return rms_radius(pp[:, :2]), ee50_radius(pp[:, :2]), int(alive.sum())

def main():
    print("=" * 104)
    print("SEQUENTIAL LENS RAY TRACE — US6141154 Example 1 (f/0.99, EFL 105.815 mm), spots at the image surface")
    print("=" * 104)
    efl, fno, m, zi, vig, ok_efl, ok_fno = gate()
    print(f"  [G-EFL] paraxial EFL = {efl:.3f} mm vs patent {EFL_PATENT} mm  ({100*abs(efl-EFL_PATENT)/EFL_PATENT:.3f}%) -> {ok_efl}")
    print(f"  [G-FNO] real-ray working f/# = {fno:.3f} vs patent f/{FNO_PATENT}  ({100*abs(fno-FNO_PATENT)/FNO_PATENT:.2f}%) -> {ok_fno}")
    print(f"  magnification m = {m:.5f} (patent {MAG_PATENT}); image plane z = {zi:.3f} mm from the S1 vertex")
    print(f"  vignetting at full field = {100*vig:.1f}% (patent Table 4 reports ~29%)")

    sysd = config_A()
    sag_img = lambda r: -(r ** 2) / (2.0 * R_PHOSPHOR)
    rows = spot_table_fwd(sysd, m)
    print("\n  spot table, flat image plane:")
    print_tab(rows, m)
    rows_c = spot_table_fwd(sysd, m, img_sag=sag_img)
    print("\n  spot table, curved image surface (R = %.0f mm):" % R_PHOSPHOR)
    print_tab(rows_c, m)

    r_axis = rows[0][2]
    ok_spot = np.isfinite(r_axis) and r_axis < 0.05          # axial RMS spot well inside the design image
    rev_rms, rev_n = reverse_check(sysd, 0.0)
    fwd_obj = r_axis / abs(m)
    ok_rev = np.isfinite(rev_rms) and abs(rev_rms - fwd_obj) / max(fwd_obj, 1e-9) < 0.5
    print(f"\n  reversibility cross-check on axis: forward spot x |1/m| = {fwd_obj:.3f} mm vs reverse-traced {rev_rms:.3f} mm "
          f"({rev_n} rays) -> {ok_rev}")

    ok = ok_efl and ok_fno and ok_spot and ok_rev
    print("\n" + "=" * 104)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} — G-EFL={ok_efl} G-FNO={ok_fno} axial-spot={ok_spot} reversibility={ok_rev}")
    print("=" * 104)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
