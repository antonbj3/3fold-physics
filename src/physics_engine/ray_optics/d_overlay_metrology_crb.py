#!/usr/bin/env python3
r"""
Diffraction-based overlay (DBO) metrology: Cramer-Rao bounds and the overlay/grating-asymmetry degeneracy.

The nuisance parameter (grating sidewall asymmetry) is exactly degenerate with the quantity of interest (overlay) at
a single wavelength; a multi-wavelength measurement lifts the degeneracy. The module quantifies the sub-nanometre CRB
floors and the economics of the lift.

================================ FORWARD MODEL (DBO / muDBO) ================================
Two stacked line gratings (bottom + top layer), common pitch P, lateral OVERLAY offset d (the quantity of interest).
Normal-incidence illumination at wavelength lam; the +1 and -1 diffraction orders are detected.  The scalar
two-grating interference model (Fraunhofer) gives each diffracted-order amplitude as the sum of the top-grating and
bottom-grating contributions, the bottom carrying (a) a vertical STACK phase psi from propagating through the
interlayer of optical thickness n*T, and (b) a LATERAL overlay phase +/- K*d with K=2pi/P (the +1 order sees +K*d,
the -1 order sees -K*d -- grating-vector sign flip):

    E_{+1} = a_t + a_b e^{ i(psi + K d) } ,   E_{-1} = a_t + a_b e^{ i(psi - K d) }
    I_{+1} = a_t^2 + a_b^2 + 2 a_t a_b cos(psi + K d)
    I_{-1} = a_t^2 + a_b^2 + 2 a_t a_b cos(psi - K d)
    ASYMMETRY  A(d) = I_{+1} - I_{-1} = -4 a_t a_b sin(psi) sin(K d)
              small-d:  A(d) ~ K_DBO * d ,   K_DBO = -4 a_t a_b sin(psi) * K            (the DBO linear law)

This reproduces the textbook DBO "swing curve": the overlay sensitivity K_DBO oscillates with sin(psi(lam)), i.e.
with wavelength / interlayer thickness -- which is why production DBO is multi-wavelength.
  [Refs: Adel et al., "Diffraction order control in overlay metrology", Proc. SPIE 5375 (2004);
   Bhattacharyya et al., "A study of overlay ... diffraction based overlay", Proc. SPIE (2013);
   the +/-1 asymmetry ~ K*overlay linearisation is the standard estimator.]

NUISANCES (the degeneracy sources):
  * grating ASYMMETRY g (sidewall-angle imbalance): the bottom grating diffracts +1 and -1 with UNEQUAL
    amplitude a_b(1+/-g) -- a physical top/bottom sidewall tilt.  To first order it injects a d-INDEPENDENT
    asymmetry offset:  A -> K_DBO*d + K_asym*g ,  K_asym = 4( a_b^2 + a_t a_b cos(psi) ).  This is the killer.
  * interlayer thickness variation T (moves psi, hence K_DBO and K_asym) -- the physical lever that makes
    K_DBO(lam), K_asym(lam) wavelength-dependent, and therefore the basis of the multi-wavelength lift.
  * intensity fluctuation (common-mode gain) -- cancels in the +/-1 asymmetry difference (declared).

================================ THE CRB LADDER ================================
Photon-limited detection: each order is a Poisson count, mean mu_k = N * I_k (N = photon-budget scale).
Fisher:  F_ij = sum_k (1/mu_k)(d mu_k/d theta_i)(d mu_k/d theta_j),  theta=(d, g[, ...]).
  P1  nuisance-free CRB (g known): sigma(d) >= 1/sqrt(F_dd) ~ 1/sqrt(N)  -- reached within 2x by the real
      asymmetry estimator (Poisson Monte Carlo, >=300 realisations).
  P2  DEGENERACY TEST: the (d,g) 2x2 Fisher at ONE wavelength is EXACTLY rank-1 (dI/dd and dI/dg are collinear
      over the +/-1 orders), so the marginalised CRB for d is infinite (an exact null); the invisible combination
      K_DBO*dd + K_asym*dg = 0 is the null direction.  Multiple wavelengths lift it because the ratio
      K_asym(lam)/K_DBO(lam) varies with lam (sin psi vs cos psi swing), making the Fisher matrix rank-2.
      The marginalised-CRB inflation is measured vs the number of wavelengths at fixed total photon budget
      (W wavelengths each get N/W photons).
  P3  sub-nm reality check: lam=500-700 nm, P~600 nm, N~1e6-1e9 -> does the certified sigma(d) reach the
      ~0.1 nm overlay class used in production metrology?

INPUT: none (closed-form forward model plus Monte-Carlo).
OUTPUT: printed gate lines + `artifacts/d_overlay_metrology_evidence.json`.
"""
import json, os, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "artifacts", "d_overlay_metrology_evidence.json")
RNG = np.random.default_rng(20260707)

# ------------------------------------------------------------------ FROZEN prereg (numbers before measuring)
PREREG = {
    "domain": "optical overlay metrology (DBO) -- overlay/asymmetry degeneracy and CRB floors",
    "P1_estimator_within": 2.0,          # real estimator var / CRB must be <= this (MC)
    "P1_slope_1_over_sqrtN": -0.5,        # log sigma(d) vs log N slope
    "P1_slope_tol": 0.03,
    "P2_single_wavelength_is_exact_null": True,   # 2x2 Fisher rank-1 -> inflation -> inf
    "P2_null_cossim_min": 0.999,          # measured null vs predicted (K_asym,-K_DBO) direction
    "P2_multiwavelength_lifts": True,     # W>=2 -> finite marginalised CRB
    "P3_target_overlay_nm": 0.1,          # industrial ~0.1 nm class
    "P3_reaches_subnm": True,             # certified sigma(d) <= a few x target at realistic N
    "linearity_max_dev_pct_at_10nm": 1.0, # A(d) vs K_DBO*d over +/-10 nm
    "C":  "single-lambda DBO = exact overlay/asymmetry degeneracy (null cos-sim ~1); multi-lambda LIFTS "
          "(finite CRB, measured lift factor); certified sigma(d) reaches ~0.1nm class at N~1e6-1e9 "
          "=> degeneracy confirmed and lifted, and the production operating point is reproduced.",
    "notC": "single-lambda already identifiable (no null), OR multi-lambda fails to lift, OR sigma(d) cannot "
            "reach the 0.1nm class at realistic photon budgets (floor mis-anchored).",
}

C_LIGHT = 299792458.0  # not used dimensionfully; lengths in nm, phases dimensionless


# ================================================================== FORWARD MODEL
class Stack:
    """DBO target: pitch P (nm), interlayer optical path n*T (nm), diffraction amplitudes a_t,a_b.
    psi(lam) = 4*pi*n*T/lam  (round trip through the interlayer)."""
    def __init__(self, P=600.0, nT=630.0, a_t=0.30, a_b=0.30):
        self.P, self.nT, self.a_t, self.a_b = P, nT, a_t, a_b
        self.K = 2.0 * np.pi / P  # grating vector, per nm

    def psi(self, lam):
        return 4.0 * np.pi * self.nT / lam

    def intensities(self, d, g, lam):
        """EXACT (nonlinear) +/-1 order intensities with overlay d (nm) and grating asymmetry g.
        Bottom grating diffracts +1/-1 with amplitude a_b(1+/-g)."""
        at, ab, K, psi = self.a_t, self.a_b, self.K, self.psi(lam)
        Ip = at**2 + (ab*(1+g))**2 + 2*at*ab*(1+g)*np.cos(psi + K*d)
        Im = at**2 + (ab*(1-g))**2 + 2*at*ab*(1-g)*np.cos(psi - K*d)
        return Ip, Im

    def K_DBO(self, lam):   # d(A)/d(d) at (0,0) linear overlay sensitivity
        return -4.0*self.a_t*self.a_b*np.sin(self.psi(lam))*self.K

    def K_asym(self, lam):  # d(A)/d(g) at (0,0) linear asymmetry coupling
        return 4.0*(self.a_b**2 + self.a_t*self.a_b*np.cos(self.psi(lam)))

    def I0(self, lam):      # mean order intensity at (0,0)
        return self.a_t**2 + self.a_b**2 + 2*self.a_t*self.a_b*np.cos(self.psi(lam))


# ================================================================== FISHER (Poisson, exact-model gradients)
def fisher_single(stack, lam, N, d0=0.0, g0=0.0, params=("d", "g")):
    """2x2 (or 1x1) Poisson Fisher for theta=params at (d0,g0), photon scale N, one wavelength.
    Gradients by central finite difference on the EXACT intensities (independent of the linear K's)."""
    hd, hg = 1e-3, 1e-5  # d in nm, g dimensionless
    def mu(d, g):
        Ip, Im = stack.intensities(d, g, lam)
        return np.array([N*Ip, N*Im])
    grads = {}
    grads["d"] = (mu(d0+hd, g0) - mu(d0-hd, g0)) / (2*hd)
    grads["g"] = (mu(d0, g0+hg) - mu(d0, g0-hg)) / (2*hg)
    mu0 = mu(d0, g0)
    n = len(params)
    F = np.zeros((n, n))
    for i, pi in enumerate(params):
        for j, pj in enumerate(params):
            F[i, j] = np.sum(grads[pi]*grads[pj]/mu0)
    return F, mu0, grads


def fisher_multi(stack, lams, N_total, split=True, d0=0.0, g0=0.0):
    """(d,g) Fisher summed over independent wavelengths. split=True -> each lam gets N_total/len(lams)
    photons (honest fixed-budget economics); split=False -> each lam gets N_total (more total light)."""
    W = len(lams)
    Nper = N_total/W if split else N_total
    F = np.zeros((2, 2))
    for lam in lams:
        Fi, _, _ = fisher_single(stack, lam, Nper, d0, g0, params=("d", "g"))
        F += Fi
    return F


def crb_d_marginalised(F):
    """CRB for d marginalising the nuisance g: (F^{-1})_dd.  Returns (variance, inflation-vs-nuisance-free)."""
    Fdd = F[0, 0]
    if Fdd <= 0:
        return np.inf, np.inf
    # regularised inverse to expose the singular (rank-1) case honestly
    det = np.linalg.det(F)
    if det <= Fdd*F[1, 1]*1e-12:  # numerically singular
        return np.inf, np.inf
    var_marg = np.linalg.inv(F)[0, 0]
    var_free = 1.0/Fdd
    return var_marg, var_marg/var_free


# ================================================================== P0  linearity validation
def validate_linearity(stack, lam, dmax=10.0):
    ds = np.linspace(-dmax, dmax, 41)
    A_exact = np.array([np.diff(stack.intensities(d, 0.0, lam))[0]*-1 for d in ds])  # Ip-Im
    A_exact = np.array([(lambda t: t[0]-t[1])(stack.intensities(d, 0.0, lam)) for d in ds])
    A_lin = stack.K_DBO(lam)*ds
    nz = np.abs(A_lin) > 1e-9
    dev = np.max(np.abs(A_exact[nz]-A_lin[nz])/np.abs(A_lin[nz]))*100
    return {"lam_nm": lam, "dmax_nm": dmax, "max_dev_pct": float(dev),
            "K_DBO_per_nm": float(stack.K_DBO(lam))}


# ================================================================== P1  nuisance-free CRB + MC estimator
def p1_nuisance_free(stack, lam, N):
    F, mu0, grads = fisher_single(stack, lam, N, params=("d",))
    Fdd = F[0, 0]
    sigma_crb = 1.0/np.sqrt(Fdd)
    return sigma_crb, Fdd, mu0


def p1_mc_estimator(stack, lam, N, n_mc=500, d_true=0.0):
    """Poisson MC: sample +/-1 counts, estimate d via the asymmetry estimator d_hat = (n+ - n-)/(N*K_DBO)."""
    Ip, Im = stack.intensities(d_true, 0.0, lam)
    mu_p, mu_m = N*Ip, N*Im
    KDBO = stack.K_DBO(lam)
    np_s = RNG.poisson(mu_p, n_mc).astype(float)
    nm_s = RNG.poisson(mu_m, n_mc).astype(float)
    d_hat = (np_s - nm_s)/(N*KDBO)
    return float(np.mean(d_hat)), float(np.std(d_hat, ddof=1))


# ================================================================== P2  masquerade + lift
def predicted_null(stack, lam):
    """The (d,g) direction annihilating the port asymmetry: K_DBO*dd + K_asym*dg = 0
       => (dd,dg) ∝ (K_asym, -K_DBO), normalised."""
    v = np.array([stack.K_asym(lam), -stack.K_DBO(lam)])
    return v/np.linalg.norm(v)


def p2_masquerade(stack, lam, N):
    F, _, _ = fisher_single(stack, lam, N, params=("d", "g"))
    w, V = np.linalg.eigh(F)               # ascending eigenvalues
    null_meas = V[:, 0]                    # smallest-eigenvalue eigenvector
    null_pred = predicted_null(stack, lam)
    cossim = float(abs(np.dot(null_meas, null_pred)))
    eig_ratio = float(w[0]/w[-1])          # ~0 -> rank-1 -> exact null
    var_marg, infl = crb_d_marginalised(F)
    return {"lam_nm": lam, "eig_min": float(w[0]), "eig_max": float(w[-1]),
            "eig_ratio_min_over_max": eig_ratio, "rank1_singular": bool(eig_ratio < 1e-8),
            "null_cossim_meas_vs_pred": cossim,
            "null_vector_pred_(d,g)": null_pred.tolist(),
            "marg_CRB_var_d": (None if not np.isfinite(var_marg) else float(var_marg)),
            "inflation_vs_nuisance_free": (None if not np.isfinite(infl) else float(infl)),
            "inflation_is_infinite": bool(not np.isfinite(infl))}


def p2_lift_curve(stack, lams_pool, N_total, Ws, split=True):
    """Marginalised sigma(d) as #wavelengths grows (fixed total photon budget). W=1 -> inf (null)."""
    rows = []
    for W in Ws:
        lams = list(lams_pool[:W])
        F = fisher_multi(stack, lams, N_total, split=split)
        var_marg, infl = crb_d_marginalised(F)
        # nuisance-free single-best-lam CRB at same TOTAL budget for reference
        w, V = np.linalg.eigh(F)
        rows.append({"W": W, "lams_nm": lams,
                     "marg_sigma_d_nm": (None if not np.isfinite(var_marg) else float(np.sqrt(var_marg))),
                     "inflation_vs_nuisance_free": (None if not np.isfinite(infl) else float(infl)),
                     "fisher_eig_min": float(w[0]), "fisher_eig_max": float(w[-1]),
                     "singular": bool(w[0]/w[-1] < 1e-8)})
    return rows


def p2_joint_mc(stack, lams, N_total, n_mc=400, d_true=0.0, g_true=0.02, split=True):
    """Joint (d,g) WLS estimator over multi-lambda asymmetries reaches the marginalised CRB;
       a single-lambda estimator that IGNORES g is BIASED by (K_asym/K_DBO)*g (the degeneracy error)."""
    W = len(lams)
    Nper = N_total/W if split else N_total
    # design: A_lam = Nper*(K_DBO(lam) d + K_asym(lam) g); WLS with per-lam variance 2*Nper*I0
    Jd = np.array([Nper*stack.K_DBO(l) for l in lams])
    Jg = np.array([Nper*stack.K_asym(l) for l in lams])
    Jm = np.vstack([Jd, Jg]).T                       # (W,2)
    var_A = np.array([2*Nper*stack.I0(l) for l in lams])
    Winv = np.diag(1.0/var_A)
    cov = np.linalg.inv(Jm.T @ Winv @ Jm)            # 2x2 estimator covariance (== asymmetry Fisher inverse)
    d_hats, d_hats_single = [], []
    KDBO0, Kasym0 = stack.K_DBO(lams[0]), stack.K_asym(lams[0])
    for _ in range(n_mc):
        A_obs = []
        for l in lams:
            Ip, Im = stack.intensities(d_true, g_true, l)
            npp = RNG.poisson(Nper*Ip); nmm = RNG.poisson(Nper*Im)
            A_obs.append(npp-nmm)
        A_obs = np.array(A_obs, float)
        theta = cov @ (Jm.T @ Winv @ A_obs)          # WLS solve
        d_hats.append(theta[0])
        # naive single-lambda estimator (ignores g): d_hat = A_0/(Nper*K_DBO_0)
        d_hats_single.append(A_obs[0]/(Nper*KDBO0))
    d_hats = np.array(d_hats); d_hats_single = np.array(d_hats_single)
    predicted_single_bias = (Kasym0/KDBO0)*g_true
    return {"g_true": g_true, "d_true": d_true,
            "joint_mean_d": float(np.mean(d_hats)), "joint_std_d": float(np.std(d_hats, ddof=1)),
            "joint_CRB_sigma_d": float(np.sqrt(cov[0, 0])),
            "joint_std_over_CRB": float(np.std(d_hats, ddof=1)/np.sqrt(cov[0, 0])),
            "joint_bias_nm": float(np.mean(d_hats)-d_true),
            "single_lam_mean_d": float(np.mean(d_hats_single)),
            "single_lam_bias_nm": float(np.mean(d_hats_single)-d_true),
            "single_lam_predicted_bias_nm": float(predicted_single_bias)}


# ================================================================== P3  sub-nm reality check
def p3_reality(stack, lams, Ns):
    rows = []
    # best single wavelength (max nuisance-free Fisher) for the nuisance-free column
    best_lam = max(lams, key=lambda l: abs(stack.K_DBO(l))/np.sqrt(stack.I0(l)))
    for N in Ns:
        s_free, Fdd, _ = p1_nuisance_free(stack, best_lam, N)     # nuisance-free, best lam
        F = fisher_multi(stack, lams, N, split=True)              # multi-lam, marginalised, split budget
        var_marg, infl = crb_d_marginalised(F)
        rows.append({"N_photons": float(N), "best_lam_nm": float(best_lam),
                     "sigma_d_nuisancefree_nm": float(s_free),
                     "sigma_d_marginalised_multiwl_nm": (None if not np.isfinite(var_marg) else float(np.sqrt(var_marg))),
                     "inflation": (None if not np.isfinite(infl) else float(infl))})
    return rows, best_lam


def loglog_slope(x, y):
    lx, ly = np.log(np.asarray(x, float)), np.log(np.asarray(y, float))
    return float(np.polyfit(lx, ly, 1)[0])


# ================================================================== MAIN
def main():
    t0 = time.time()
    stack = Stack(P=600.0, nT=630.0, a_t=0.30, a_b=0.30)   # nT chosen so psi sweeps well over 500-700nm
    lam_c = 600.0

    # wavelength pool for lift/reality (spread so sin psi / cos psi swing decorrelates K_DBO,K_asym)
    lam_pool = [500.0, 540.0, 580.0, 620.0, 660.0, 700.0, 520.0, 640.0]
    # report per-lam couplings (verify none accidentally dead in the working set)
    coupling = [{"lam_nm": l, "psi_rad": float(stack.psi(l) % (2*np.pi)),
                 "sin_psi": float(np.sin(stack.psi(l))), "cos_psi": float(np.cos(stack.psi(l))),
                 "K_DBO_per_nm": float(stack.K_DBO(l)), "K_asym": float(stack.K_asym(l)),
                 "ratio_Kasym_over_KDBO_nm": float(stack.K_asym(l)/stack.K_DBO(l))} for l in lam_pool]

    out = {"cell": "d_overlay_metrology_crb",
           "family": "lithography metrology (diffraction-based overlay)",
           "prereg": PREREG, "forward_model": {
               "P_nm": stack.P, "nT_nm": stack.nT, "a_t": stack.a_t, "a_b": stack.a_b,
               "K_grating_per_nm": float(stack.K), "wavelength_pool_couplings": coupling}}

    # ---------- P0 linearity ----------
    lin = [validate_linearity(stack, l, 10.0) for l in (500.0, 600.0, 700.0)]
    lin_maxdev = max(r["max_dev_pct"] for r in lin)
    out["P0_linearity"] = {"per_lam": lin, "max_dev_pct_any": lin_maxdev,
                           "PASS": bool(lin_maxdev < PREREG["linearity_max_dev_pct_at_10nm"])}

    # ---------- P1 nuisance-free CRB + MC ----------
    # pick a sensitive wavelength for P1 (max |K_DBO|/sqrt(I0))
    lam1 = max(lam_pool, key=lambda l: abs(stack.K_DBO(l))/np.sqrt(stack.I0(l)))
    Ns_crb = [1e6, 1e7, 1e8, 1e9]
    sig_free = [p1_nuisance_free(stack, lam1, N)[0] for N in Ns_crb]
    slope = loglog_slope(Ns_crb, sig_free)
    # MC at one N to confirm estimator reaches CRB
    N_mc = 1e8
    s_crb_mc, _, _ = p1_nuisance_free(stack, lam1, N_mc)
    mc_mean, mc_std = p1_mc_estimator(stack, lam1, N_mc, n_mc=500)
    out["P1_nuisance_free"] = {
        "lam_nm": lam1, "N_grid": Ns_crb, "sigma_d_nm": [float(s) for s in sig_free],
        "slope_log_sigma_vs_log_N": slope,
        "slope_is_-0.5": bool(abs(slope - PREREG["P1_slope_1_over_sqrtN"]) < PREREG["P1_slope_tol"]),
        "MC": {"N": N_mc, "n_mc": 500, "crb_sigma_nm": s_crb_mc, "mc_std_nm": mc_std,
               "mc_bias_nm": mc_mean, "mc_std_over_crb": mc_std/s_crb_mc,
               "reaches_within_2x": bool(mc_std/s_crb_mc <= PREREG["P1_estimator_within"])}}

    # ---------- P2 masquerade (single wavelength) ----------
    masq = p2_masquerade(stack, lam1, 1e8)
    out["P2_masquerade_single_wavelength"] = masq

    # ---------- P2 lift curve (multi-wavelength, fixed total budget) ----------
    Ws = [1, 2, 3, 4, 5, 6]
    lift_split = p2_lift_curve(stack, lam_pool, 1e8, Ws, split=True)
    lift_nosplit = p2_lift_curve(stack, lam_pool, 1e8, Ws, split=False)
    # lift factor: ratio of (W=2 finite) to the W>=2 asymptote; and "does W>=2 become finite"
    finite_W = [r["W"] for r in lift_split if r["marg_sigma_d_nm"] is not None]
    out["P2_lift"] = {
        "fixed_total_budget_split": lift_split,
        "per_lambda_fixed_budget": lift_nosplit,
        "W1_singular_null": bool(lift_split[0]["singular"]),
        "W2plus_finite": bool(all(r["marg_sigma_d_nm"] is not None for r in lift_split if r["W"] >= 2)),
        "first_finite_W": (min(finite_W) if finite_W else None)}

    # ---------- P2 joint-MC (estimator reaches marginalised CRB; single-lam biased by masquerade) ----------
    lams4 = [500.0, 580.0, 620.0, 700.0]
    jmc = p2_joint_mc(stack, lams4, 1e8, n_mc=400, d_true=0.0, g_true=0.02, split=True)
    jmc["joint_reaches_CRB_within_2x"] = bool(jmc["joint_std_over_CRB"] <= 2.0)
    jmc["single_lam_masquerade_bias_confirmed"] = bool(
        abs(jmc["single_lam_bias_nm"] - jmc["single_lam_predicted_bias_nm"]) <
        0.15*abs(jmc["single_lam_predicted_bias_nm"]) + 1e-3)
    out["P2_joint_MC"] = jmc

    # ---------- P3 reality ----------
    p3_rows, best_lam = p3_reality(stack, lams4, [1e6, 1e7, 1e8, 1e9])
    # does the marginalised (realistic, multi-wl) sigma reach the 0.1nm class?
    reach = {r["N_photons"]: r["sigma_d_marginalised_multiwl_nm"] for r in p3_rows}
    n_reach_marg = [N for N, s in reach.items() if s is not None and s <= PREREG["P3_target_overlay_nm"]]
    reach_free = {r["N_photons"]: r["sigma_d_nuisancefree_nm"] for r in p3_rows}
    n_reach_free = [N for N, s in reach_free.items() if s <= PREREG["P3_target_overlay_nm"]]
    out["P3_reality_check"] = {
        "target_overlay_nm": PREREG["P3_target_overlay_nm"], "best_lam_nm": float(best_lam),
        "table": p3_rows,
        "nuisance_free_reaches_0p1nm_at_N": (min(n_reach_free) if n_reach_free else None),
        "marginalised_multiwl_reaches_0p1nm_at_N": (min(n_reach_marg) if n_reach_marg else None),
        "reaches_industrial_class": bool(n_reach_marg or n_reach_free)}

    # ---------- VERDICT ----------
    v_lin = out["P0_linearity"]["PASS"]
    v_p1 = out["P1_nuisance_free"]["slope_is_-0.5"] and out["P1_nuisance_free"]["MC"]["reaches_within_2x"]
    v_null = masq["rank1_singular"] and masq["inflation_is_infinite"] and \
        masq["null_cossim_meas_vs_pred"] >= PREREG["P2_null_cossim_min"]
    v_lift = out["P2_lift"]["W1_singular_null"] and out["P2_lift"]["W2plus_finite"]
    v_joint = jmc["joint_reaches_CRB_within_2x"] and jmc["single_lam_masquerade_bias_confirmed"]
    v_p3 = out["P3_reality_check"]["reaches_industrial_class"]
    masquerade_confirmed = bool(v_null and v_lift and v_joint)
    out["VERDICT"] = {
        "P0_linearity_ok": bool(v_lin),
        "P1_nuisance_free_CRB_1_over_sqrtN_and_estimator_efficient": bool(v_p1),
        "P2_single_lambda_EXACT_NULL_(masquerade)": bool(v_null),
        "P2_null_cossim": masq["null_cossim_meas_vs_pred"],
        "P2_multiwavelength_LIFTS": bool(v_lift),
        "P2_joint_estimator_efficient_and_single_lam_biased": bool(v_joint),
        "P3_reaches_industrial_0p1nm_class": bool(v_p3),
        "MASQUERADE_LAW_4TH_DOMAIN_CONFIRMED": masquerade_confirmed,
        "VERDICT": ("CONFIRMED" if (masquerade_confirmed and v_p1 and v_p3 and v_lin) else "REFUTED/PARTIAL"),
    }
    out["wall_time_s"] = round(time.time()-t0, 2)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)

    # ---------------------------------------------------------------- print
    P = print
    P("="*104)
    P("DBO OVERLAY METROLOGY -- overlay/asymmetry degeneracy + sub-nm CRB floors")
    P("="*104)
    P(f"forward model: two-grating Fraunhofer  A(d)=-4 a_t a_b sin(psi) sin(Kd) ~ K_DBO*d ;  P={stack.P}nm nT={stack.nT}nm")
    P(f"\n[P0] linearity of A(d) vs K_DBO*d over +/-10nm: max deviation {lin_maxdev:.3f}%  "
      f"[{'PASS' if v_lin else 'FAIL'}]")
    P("\n[P1] NUISANCE-FREE CRB  (lam={:.0f}nm):".format(lam1))
    for N, s in zip(Ns_crb, sig_free):
        P(f"     N={N:.0e}  sigma(d) = {s:.4e} nm")
    P(f"     slope log sigma vs log N = {slope:+.4f}  (predict -0.5)   [{'PASS' if out['P1_nuisance_free']['slope_is_-0.5'] else 'FAIL'}]")
    mc = out["P1_nuisance_free"]["MC"]
    P(f"     MC(N={N_mc:.0e},500): estimator std={mc['mc_std_nm']:.4e}nm  CRB={mc['crb_sigma_nm']:.4e}nm  "
      f"ratio={mc['mc_std_over_crb']:.3f}  [{'PASS' if mc['reaches_within_2x'] else 'FAIL'}]")
    P("\n[P2] DEGENERACY (single wavelength, the killer):")
    P(f"     2x2 (d,g) Fisher eig_min/eig_max = {masq['eig_ratio_min_over_max']:.3e}  -> rank-1 SINGULAR={masq['rank1_singular']}")
    P(f"     marginalised CRB inflation = {'INFINITE (exact null)' if masq['inflation_is_infinite'] else masq['inflation_vs_nuisance_free']}")
    P(f"     null direction (d,g) predicted={np.round(masq['null_vector_pred_(d,g)'],4)}  "
      f"cos-sim(measured,predicted)={masq['null_cossim_meas_vs_pred']:.6f}  [{'PASS' if v_null else 'FAIL'}]")
    P("\n[P2] MULTI-WAVELENGTH LIFT (fixed TOTAL photon budget N=1e8, split N/W each):")
    P(f"     {'W':>2} {'lams(nm)':30} {'marg_sigma_d(nm)':>18} {'inflation':>12} {'singular':>9}")
    for r in lift_split:
        s = "inf" if r["marg_sigma_d_nm"] is None else f"{r['marg_sigma_d_nm']:.4e}"
        inf = "inf" if r["inflation_vs_nuisance_free"] is None else f"{r['inflation_vs_nuisance_free']:.2e}"
        P(f"     {r['W']:>2} {str([int(x) for x in r['lams_nm']]):30} {s:>18} {inf:>12} {str(r['singular']):>9}")
    P(f"     W=1 singular null={out['P2_lift']['W1_singular_null']}  W>=2 finite={out['P2_lift']['W2plus_finite']}  "
      f"[{'PASS' if v_lift else 'FAIL'}]")
    P("\n[P2] JOINT MC (degeneracy error + efficient joint estimator), g_true={:.3f}:".format(jmc["g_true"]))
    P(f"     single-lambda estimator (ignores g): bias={jmc['single_lam_bias_nm']:+.4f}nm  "
      f"predicted={jmc['single_lam_predicted_bias_nm']:+.4f}nm  (the degeneracy bias)")
    P(f"     joint multi-lambda estimator: bias={jmc['joint_bias_nm']:+.4f}nm  std={jmc['joint_std_d']:.4e}nm  "
      f"CRB={jmc['joint_CRB_sigma_d']:.4e}nm  ratio={jmc['joint_std_over_CRB']:.3f}  [{'PASS' if v_joint else 'FAIL'}]")
    P("\n[P3] SUB-NM REALITY CHECK (lam 500-700nm, P=600nm):")
    P(f"     {'N_photons':>11} {'sigma_nuis-free(nm)':>20} {'sigma_marg_multiwl(nm)':>24} {'inflation':>10}")
    for r in p3_rows:
        sm = "inf" if r["sigma_d_marginalised_multiwl_nm"] is None else f"{r['sigma_d_marginalised_multiwl_nm']:.4e}"
        inf = "inf" if r["inflation"] is None else f"{r['inflation']:.2f}"
        P(f"     {r['N_photons']:>11.0e} {r['sigma_d_nuisancefree_nm']:>20.4e} {sm:>24} {inf:>10}")
    P(f"     nuisance-free reaches 0.1nm at N>={out['P3_reality_check']['nuisance_free_reaches_0p1nm_at_N']}  "
      f"| marginalised multi-wl reaches 0.1nm at N>={out['P3_reality_check']['marginalised_multiwl_reaches_0p1nm_at_N']}")
    P("-"*104)
    for k, val in out["VERDICT"].items():
        if isinstance(val, bool):
            P(f"  [{'PASS' if val else 'FAIL'}] {k}")
        else:
            P(f"         {k} = {val}")
    P(f"\n[EMIT] {OUT}   ({out['wall_time_s']}s)")


if __name__ == "__main__":
    main()
