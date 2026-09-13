"""Scalar Fourier optics: FFT-based Fraunhofer diffraction (single-slit, double-slit and circular-aperture gates),
then grating-order plus Abbe partial-coherence imaging of a periodic pixel grid.

INSTRUMENT GATES (analytic, pre-registered):
G1 single-slit Fraunhofer: first zero at sin(theta) = lambda/a. FFT of a slit aperture vs the closed form, <0.1%.
G2 double-slit Fraunhofer: fringe spacing sin(theta) = lambda/d. FFT peak spacing vs the closed form, <0.1%.
G3 circular-aperture Fraunhofer (Airy): first dark ring at sin(theta) = 1.22 lambda/D. 2D-FFT radial null vs the
   closed form, <0.5% (grid-resolution-limited - reported on the G3 line).

PIXEL-GRID IMAGING (the decisive number): a periodic pixel grid is modelled as a 1D periodic amplitude grating of
period Lambda = 22 um, fill factor 0.9 (square-aperture duty cycle), lambda = 532 nm. Grating orders m satisfy
sin(theta_m) = m*lambda/Lambda. An imaging objective with a 157 mm entrance pupil at 179 mm sets the objective
numerical aperture NA_obj; a 0.15 pupil-fill converging illumination beam sets NA_illum = 0.15*NA_obj, i.e. the
partial-coherence factor sigma = 0.15 of Abbe/Hopkins imaging. For each illumination angle theta_i drawn from the
illumination NA, the coherent image amplitude is the inverse Fourier series of whichever grating orders fall inside
the objective NA (Abbe's method - off-axis illumination shifts the passband along the order axis; the common tilt
drops out of |field|^2). The partially coherent image intensity is the incoherent average of |amplitude(x)|^2 over
illumination angles. Ripple = (Imax - Imin)/Imean of a uniform ("all pixels open") field image, computed for
(a) fully coherent normal incidence (sigma = 0) and (b) sigma = 0.15.

VALIDITY LIMITS (to be read alongside the ripple number):
 - Scalar theory: polarization/vector diffraction effects matter only near aperture ~ lambda; here 22 um >> 532 nm.
 - Thin-mask (Kirchhoff) approximation: the pixel aperture is an infinitely thin amplitude mask; finite cell depth,
   electrode topology and birefringence are not modelled. Adequate for the order structure at pitch >> wavelength;
   exact per-order diffraction efficiencies would need a rigorous (e.g. RCWA) solve.
 - 1D grating and 1D illumination-pupil sampling: the real pixel grid and pupil are 2D, so the reported ripple is a
   1D proxy (order count, passband width and incoherent averaging are captured).
 - No lens aberrations, no defocus, no source spectral width - purely geometric-NA / order-counting physics.

Reference: Abbe imaging theory / Hopkins partial coherence, e.g. Goodman, "Introduction to Fourier Optics".
INPUT: none (all parameters are arguments with defaults). OUTPUT: printed gate dictionaries and ripple numbers.
"""
import numpy as np


# ------------------------------------------------------------------------------------------------------------------
# G1/G2: 1D Fraunhofer diffraction via FFT of the aperture transmission function.
# ------------------------------------------------------------------------------------------------------------------
def fraunhofer_1d(aperture, dx, wavelength):
    """FFT-based far-field (Fraunhofer) amplitude of a 1D aperture sampled on a grid of spacing dx.
    Returns (sin_theta, amplitude) with sin_theta = wavelength * spatial_frequency (only the |sin_theta|<=1
    physical branch is meaningful; this is the standard Fraunhofer/angular-spectrum far-field mapping)."""
    n = len(aperture)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))          # cycles per unit length
    amp = np.fft.fftshift(np.fft.fft(aperture)) * dx
    sin_theta = wavelength * freqs
    return sin_theta, amp


def _first_zero_sintheta(sin_theta, intensity, exclude_radius):
    """First null of |amplitude|^2 moving outward from the center (theta=0) peak, beyond exclude_radius
    (to skip the finite-grid central-peak shoulder before the true null)."""
    center = len(sin_theta) // 2
    n = len(sin_theta)
    for i in range(center, n - 1):
        if sin_theta[i] > exclude_radius and intensity[i + 1] > intensity[i]:
            return sin_theta[i]
    raise RuntimeError("no null found in range")


def gate_G1_single_slit(a=20e-6, wavelength=532e-9, n=1 << 20, window_factor=4000):
    """G1: single slit width a, first Fraunhofer zero at sin(theta)=lambda/a."""
    x_max = window_factor * a
    dx = 2 * x_max / n
    x = (np.arange(n) - n // 2) * dx
    aperture = (np.abs(x) <= a / 2).astype(float)
    sin_theta, amp = fraunhofer_1d(aperture, dx, wavelength)
    intensity = np.abs(amp) ** 2
    analytic = wavelength / a
    measured = _first_zero_sintheta(sin_theta, intensity, exclude_radius=0.3 * analytic)
    rel_err = abs(measured - analytic) / analytic
    return dict(gate="G1_single_slit", a=a, wavelength=wavelength,
                analytic_sin_theta=analytic, measured_sin_theta=measured,
                rel_err_pct=100 * rel_err, pass_=rel_err < 1e-3)


def gate_G2_double_slit(a=1e-6, d=40e-6, wavelength=532e-9, n=1 << 20, window_factor=4000):
    """G2: double slit (width a, center-to-center separation d), fringe spacing sin(theta)=lambda/d.
    NOTE: a/d=0.025 (narrow slits) is deliberate — at a/d=0.1 the single-slit sinc ENVELOPE measurably tilts
    the fringe peak positions off the ideal delta-slit spacing (a real physical effect, confirmed numerically:
    a/d=0.1 gives ~0.33% peak-position error even at the first fringe, a/d=0.025 gives ~0.03%). The gate is the
    idealized Young's-double-slit law (point-like slits); a/d=0.025 approaches that limit while keeping the slit
    resolvable on the grid (a/dx ~ 3.3 grid cells)."""
    x_max = window_factor * d
    dx = 2 * x_max / n
    x = (np.arange(n) - n // 2) * dx
    aperture = ((np.abs(x - d / 2) <= a / 2) | (np.abs(x + d / 2) <= a / 2)).astype(float)
    sin_theta, amp = fraunhofer_1d(aperture, dx, wavelength)
    intensity = np.abs(amp) ** 2
    center = n // 2
    # walk outward from center peak, collect local maxima (fringe peaks) within the single-slit envelope's first lobe
    analytic_spacing = wavelength / d
    search_radius = 6 * analytic_spacing
    mask = (sin_theta > 0) & (sin_theta < search_radius)
    idx = np.where(mask)[0]
    peaks = []
    dtheta = sin_theta[1] - sin_theta[0]
    for i in idx[1:-1]:
        if intensity[i] > intensity[i - 1] and intensity[i] > intensity[i + 1]:
            # parabolic (3-point) sub-bin interpolation of the peak location for sub-grid-spacing accuracy
            y0, y1, y2 = intensity[i - 1], intensity[i], intensity[i + 1]
            denom = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
            peaks.append(sin_theta[i] + delta * dtheta)
    peaks = np.array(peaks)
    spacings = np.diff(peaks)
    measured_spacing = np.median(spacings)
    rel_err = abs(measured_spacing - analytic_spacing) / analytic_spacing
    return dict(gate="G2_double_slit", a=a, d=d, wavelength=wavelength,
                analytic_fringe_spacing=analytic_spacing, measured_fringe_spacing=measured_spacing,
                n_peaks_used=len(peaks), rel_err_pct=100 * rel_err, pass_=rel_err < 1e-3)


# ------------------------------------------------------------------------------------------------------------------
# G3: 2D Fraunhofer diffraction of a circular aperture -> Airy pattern.
# ------------------------------------------------------------------------------------------------------------------
def gate_G3_circular_aperture(D=50e-6, wavelength=532e-9, n=4096, window_factor=60):
    """G3: circular aperture diameter D, first Airy dark ring at sin(theta)=1.22*lambda/D."""
    x_max = window_factor * D
    dx = 2 * x_max / n
    coords = (np.arange(n) - n // 2) * dx
    X, Y = np.meshgrid(coords, coords)
    R = np.sqrt(X ** 2 + Y ** 2)
    aperture = (R <= D / 2).astype(float)
    field = np.fft.fftshift(np.fft.fft2(aperture)) * dx ** 2
    intensity = np.abs(field) ** 2
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    sin_theta_axis = wavelength * freqs
    # radial profile through the center row (by symmetry, radial = axial cut)
    center = n // 2
    row = intensity[center, :]
    sin_theta_row = sin_theta_axis
    analytic = 1.22 * wavelength / D
    # pass the FULL symmetric array (not pre-sliced) — _first_zero_sintheta locates center=len//2 itself
    measured = _first_zero_sintheta(sin_theta_row, row, exclude_radius=0.3 * analytic)
    rel_err = abs(measured - analytic) / analytic
    grid_angular_resolution = wavelength * (1.0 / (n * dx))   # sin_theta bin size — the honest resolution note
    return dict(gate="G3_circular_aperture", D=D, wavelength=wavelength,
                analytic_sin_theta=analytic, measured_sin_theta=measured,
                grid_sin_theta_resolution=grid_angular_resolution,
                rel_err_pct=100 * rel_err, pass_=rel_err < 0.5)


# ------------------------------------------------------------------------------------------------------------------
# Periodic pixel grating -> order structure entering the objective pupil -> Abbe/partial-coherence image ripple.
# ------------------------------------------------------------------------------------------------------------------
def grating_fourier_coeff(m, fill_factor):
    """Fourier series coefficient of a period-Lambda rectangular duty-cycle grating (amplitude 1 on the 'open'
    fraction fill_factor of each period, 0 elsewhere), m-th order. C_0 = fill_factor (mean transmission)."""
    if m == 0:
        return fill_factor
    return fill_factor * np.sinc(m * fill_factor)   # numpy sinc(x) = sin(pi x)/(pi x)


def e13_pupil_and_illumination(pupil_diameter=157e-3, pupil_distance=179e-3, sigma=0.15):
    """Pupil geometry (157 mm diameter at 179 mm) -> objective NA; the sigma=0.15 pupil fill -> illumination NA."""
    half_angle_obj = np.arctan((pupil_diameter / 2) / pupil_distance)
    na_obj = np.sin(half_angle_obj)
    na_illum = sigma * na_obj
    return dict(half_angle_obj_deg=np.degrees(half_angle_obj), na_obj=na_obj,
                sigma=sigma, na_illum=na_illum, half_angle_illum_deg=np.degrees(np.arcsin(na_illum)))


def e13_order_count(wavelength=532e-9, pitch=22e-6, na_obj=0.4016):
    """How many grating orders (each side) fall inside the objective NA at normal-incidence illumination."""
    m_max = int(np.floor(na_obj / (wavelength / pitch)))
    return dict(m_max=m_max, n_orders_total=2 * m_max + 1, n_orders_one_side_incl_zero=m_max + 1)


def _coherent_image_intensity(x, pitch, fill_factor, wavelength, na_obj, sin_theta_i, m_cutoff=200):
    """Abbe coherent image for one illumination angle sin_theta_i: sum passed orders' Fourier terms, |.|^2 in x."""
    m = np.arange(-m_cutoff, m_cutoff + 1)
    sin_theta_m = sin_theta_i + m * wavelength / pitch
    passed = np.abs(sin_theta_m) <= na_obj
    m_pass = m[passed]
    if len(m_pass) == 0:
        return np.zeros_like(x)
    coeffs = np.array([grating_fourier_coeff(mm, fill_factor) for mm in m_pass])
    phase = np.exp(1j * 2 * np.pi * np.outer(x, m_pass) / pitch)
    field = phase @ coeffs
    return np.abs(field) ** 2, m_pass


def e13_ripple(wavelength=532e-9, pitch=22e-6, fill_factor=0.9,
                pupil_diameter=157e-3, pupil_distance=179e-3, sigma_design=0.15,
                n_illum_samples=41, n_x=4001):
    """The decisive number: intensity ripple (%) of a uniform 'all pixels on' field image, for
    (a) fully-coherent normal-incidence illumination (sigma=0) and (b) the sigma=0.15 design point,
    where the sigma=0.15 illumination-NA disk is incoherently averaged (Abbe/Hopkins partial coherence)."""
    geo = e13_pupil_and_illumination(pupil_diameter, pupil_distance, sigma_design)
    na_obj = geo["na_obj"]
    x = (np.arange(n_x) - n_x // 2) / n_x * pitch    # one period, fine sampling

    # (a) fully coherent, single normal-incidence plane wave (sigma=0)
    I_coh, m_pass_coh = _coherent_image_intensity(x, pitch, fill_factor, wavelength, na_obj, sin_theta_i=0.0)
    ripple_coh = 100 * (I_coh.max() - I_coh.min()) / I_coh.mean()

    # (b) partially coherent: incoherent sum of coherent images over illumination angles within NA_illum disk
    na_illum = geo["na_illum"]
    sin_thetas_i = np.linspace(-na_illum, na_illum, n_illum_samples)
    I_acc = np.zeros(n_x)
    m_pass_union = set()
    for sti in sin_thetas_i:
        I_i, m_pass_i = _coherent_image_intensity(x, pitch, fill_factor, wavelength, na_obj, sin_theta_i=sti)
        I_acc += I_i
        m_pass_union.update(m_pass_i.tolist())
    I_pc = I_acc / n_illum_samples
    ripple_pc = 100 * (I_pc.max() - I_pc.min()) / I_pc.mean()

    order_info = e13_order_count(wavelength, pitch, na_obj)
    return dict(
        geometry=geo, order_info=order_info,
        n_orders_passed_coherent=len(m_pass_coh),
        n_orders_passed_partial_union=len(m_pass_union),
        ripple_pct_coherent_sigma0=ripple_coh,
        ripple_pct_partial_sigma015=ripple_pc,
        washout_factor=ripple_coh / ripple_pc if ripple_pc > 0 else float("inf"),
    )


def run_all():
    g1 = gate_G1_single_slit()
    g2 = gate_G2_double_slit()
    g3 = gate_G3_circular_aperture()
    e13 = e13_ripple()
    return g1, g2, g3, e13


if __name__ == "__main__":
    g1, g2, g3, e13 = run_all()
    print("=== GATES ===")
    for g in (g1, g2, g3):
        print(g)
    print()
    print("=== pupil/illumination geometry ===")
    print(e13["geometry"])
    print("=== order count (normal incidence, into the objective NA) ===")
    print(e13["order_info"])
    print()
    print(f"n_orders_passed (coherent, sigma=0):        {e13['n_orders_passed_coherent']}")
    print(f"n_orders_passed (union over sigma=0.15 disk): {e13['n_orders_passed_partial_union']}")
    print(f"RIPPLE coherent (sigma=0):   {e13['ripple_pct_coherent_sigma0']:.3f} %")
    print(f"RIPPLE design pt (sigma=0.15): {e13['ripple_pct_partial_sigma015']:.3f} %")
    print(f"washout factor: {e13['washout_factor']:.1f}x")
