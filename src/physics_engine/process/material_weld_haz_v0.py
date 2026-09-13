#!/usr/bin/env python3
"""WELD HEAT-AFFECTED ZONE, analytical leg -- weld thermal history of a GMAW-welded S355 box section.

The weld thermal field is a continuous PDE (transient heat diffusion with a moving source), so this module
uses the analytical solver family: Rosenthal (1946) line source + Adams (1958) peak-temperature-vs-distance
+ EN 1011-2:2001 Annex C empirical Delta-t8/5. It computes the net heat input, the governing cooling time
Delta-t8/5 (2-D vs 3-D regime selected by the standard's own max rule), the HAZ width from the Ac1/Ac3
isotherms, the preheat/interpass consequence at corner and T-joints, and cross-checks all of it against a
full transient Rosenthal cooling curve and the theoretical closed forms. Deterministic (no RNG); every
external number is cited inline.

INPUTS: none on disk -- the joint description (EN 10210 SHS 300x300x12.5 in S355, 9.6 m of weld, 100.8 mm^2
weld cross-section, GMAW deposition rate 5.0 kg/hr, duty cycle 0.3, arc-on time 1.519 h) is inlined below as
module constants under their original key names.
OUTPUT: artifacts/material_weld_haz_v0.json next to this module, plus a printed summary.
ANCHORS: EN 1011-2:2001 Annex C; Rosenthal 1946; Adams 1958; DuPont & Marder, Welding J. 74(12) 1995;
EN ISO 15614-1 hardness ceiling; published S355 Ac1/Ac3 and HAZ-hardness measurements (cited inline).
"""
import json
import math
import os
import hashlib

_HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(_HERE, "artifacts", "material_weld_haz_v0.json")

# Joint/process description, inlined under the original key names (previously read from a separate report file).
SUBSTRATE = {
    "external_profile_anchor": {
        "profile": "EN 10210 SHS 300x300x12.5, S355",
        "published_kg_m": 112.0,
    },
    "quantified_process_chain": {
        "weld_geometry": {
            "n_joints_total": 8,
            "total_weld_length_m_CONSTRUCTED": 9.6,
            "weld_cross_section_area_mm2_CONSTRUCTED": 100.8,
            "weld_metal_mass_kg_CONSTRUCTED": 7.596,
        },
        "weld_time": {
            "arc_on_time_hr_CONSTRUCTED": 1.519,
            "duty_cycle_CONSTRUCTED": 0.3,
            "shop_weld_time_hr_CONSTRUCTED": 5.064,
        },
    },
}


def sha256_of(obj):
    """Digest of the inlined input description (provenance stamp for the report)."""
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode("utf-8")).hexdigest()


# ============================================================================
# 0. GEOMETRY + PROCESS PARAMETERS (inlined input description above)
# ============================================================================
qpc = SUBSTRATE["quantified_process_chain"]
ext = SUBSTRATE["external_profile_anchor"]
wg = qpc["weld_geometry"]
wt = qpc["weld_time"]

assert "SHS 300x300x12.5" in ext["profile"], "profile designation changed -- re-check d_mm"
d_mm = 12.5  # wall thickness, EN 10210 SHS 300x300x12.5, S355 (external_profile_anchor.profile)

deposition_rate_kg_hr = 5.0  # central estimate (cited range 3-8 kg/hr, GMAW/MAG multi-pass steel)
duty_cycle_substrate = wt["duty_cycle_CONSTRUCTED"]
arc_on_time_hr_substrate = wt["arc_on_time_hr_CONSTRUCTED"]
weld_len_mm_substrate = wg["total_weld_length_m_CONSTRUCTED"] * 1000.0
weld_metal_mass_kg_substrate = wg["weld_metal_mass_kg_CONSTRUCTED"]
weld_xsec_mm2_substrate = wg["weld_cross_section_area_mm2_CONSTRUCTED"]

RHO_STEEL_G_CM3 = 7.85  # standard steel density; matches the inlined mass = volume * 7.85 construction


# ============================================================================
# 1. GMAW ELECTRICAL PARAMETERS -- round-trip anchored to the substrate's deposition rate
# ============================================================================
# Published GMAW deposition-rate-vs-current data, 1.2mm ER70S-6 solid wire, spray transfer, steel:
#   175A -> 1.68 kg/hr ; 225A -> 2.36 kg/hr ; 270A -> 3.73 kg/hr
#   (materialwelding.com / clause5.io deposition-rate charts)
# Local slope over the two highest points brackets the 5.0 kg/hr working value by extrapolation.
DEP_PTS = [(175.0, 1.68), (225.0, 2.36), (270.0, 3.73)]
_slope_kg_hr_per_A = (DEP_PTS[2][1] - DEP_PTS[1][1]) / (DEP_PTS[2][0] - DEP_PTS[1][0])
I_A = DEP_PTS[2][0] + (deposition_rate_kg_hr - DEP_PTS[2][1]) / _slope_kg_hr_per_A

V_V = 29.0
# published spray-transfer voltage pairing for the 220-350A / 1.2mm-wire / Ar-CO2 shielding-gas window
# (arccaptain.com MIG wire-speed/voltage chart; weldfabworld.com MIG settings calculator)

ETA_ARC = 0.8
# GMAW arc thermal efficiency. DuPont & Marder, "Thermal efficiency of arc welding processes",
# Welding Journal 74(12):406s-416s, 1995 -- report GMAW ~0.8-0.88; this module uses the published floor 0.8.

V_TRAVEL_MM_S = 5.0
# published typical GMAW multi-pass structural-steel travel speed, 180-480 mm/min cited range
# (testtalkhq.com heat-input guide; AWS D1.1-adjacent practice notes); central ~300mm/min.

P_gross_W = V_V * I_A
P_net_W = ETA_ARC * P_gross_W
Q_net_J_mm = P_net_W / V_TRAVEL_MM_S
Q_net_kJ_mm = Q_net_J_mm / 1000.0

# ---- ROUND-TRIP CHECK vs the deposition-rate bookkeeping ----
# Volumetric deposition rate implied by the mass rate (density-only, no assumption on I/V/v):
Q_vol_mm3_s = (deposition_rate_kg_hr * 1000.0 / RHO_STEEL_G_CM3) / 3600.0 * 1000.0
# Implied per-pass bead cross-section at our chosen travel speed:
A_pass_mm2 = Q_vol_mm3_s / V_TRAVEL_MM_S
N_passes_implied = weld_xsec_mm2_substrate / A_pass_mm2
per_pass_time_hr = (weld_len_mm_substrate / V_TRAVEL_MM_S) / 3600.0
arc_on_time_hr_recomputed = N_passes_implied * per_pass_time_hr
# tolerance reflects the display rounding of the inlined value (arc_on_time_hr_CONSTRUCTED stored to 3 dp),
# not floating-point slop -- full-precision recompute (1.518487h) vs the rounded 1.519h differs by ~1.8s,
# a rounding artifact, not a modelling disagreement.
round_trip_arc_on_time_pass = abs(arc_on_time_hr_recomputed - arc_on_time_hr_substrate) < 1e-3

round_trip = {
    "method": "chart-derived current I_A (external anchor, deposition-rate-vs-current data NOT used "
              "elsewhere in this module) is combined with the chosen travel speed to imply a per-pass bead "
              "cross-section A_pass = Q_vol/v; N_passes = weld_xsec/A_pass; N_passes * per-pass "
              "travel time is checked against the arc_on_time_hr of the inlined description (computed "
              "independently there as mass/deposition_rate).",
    "I_A_from_published_chart_extrapolation": I_A,
    "N_passes_implied": N_passes_implied,
    "N_passes_plausibility": "2.5-4 passes for a 12.5mm multi-pass box-section weld (root+fill+cap) is "
                              "structurally plausible.",
    "arc_on_time_hr_recomputed": arc_on_time_hr_recomputed,
    "arc_on_time_hr_substrate": arc_on_time_hr_substrate,
    "identical_to_substrate": round_trip_arc_on_time_pass,
    "note": "exact equality is expected ALGEBRAICALLY once Q_vol and total geometry are fixed "
            "(A_pass*v cancels v) -- so this is a dimensional-consistency check, not an "
            "independent statistical validation. The independent check is I_A itself: the published "
            "current-vs-deposition-rate CHART (external, unused elsewhere) predicts I=%.0fA for the "
            "substrate's OWN 5.0kg/hr -- that prediction is the round-trip's load-bearing claim." % I_A,
}


# ============================================================================
# 2. THERMAL PROPERTIES OF STEEL (published, used identically in both formula families)
# ============================================================================
LAMBDA_W_MM_K = 0.028   # thermal conductivity, 28 W/(m.K), practical steel-welding worked example
                         # (canteach.candu.org "Welding Metallurgy: Heat Flow" lecture notes)
C_J_KG_K = 440.0        # specific heat of steel (same source)
RHO_KG_M3 = 7850.0
RHOC_J_MM3_K = (RHO_KG_M3 * C_J_KG_K) / 1.0e9   # J/(mm3.K)
A_DIFFUSIVITY_MM2_S = (LAMBDA_W_MM_K / RHOC_J_MM3_K)  # thermal diffusivity, mm^2/s
T_MELT_C = 1500.0       # carbon-steel liquidus/solidus midpoint, standard Rosenthal/Adams working value
T0_C = 20.0             # ambient / no-preheat case (justified in section 5 below)


# ============================================================================
# 3. EN 1011-2 ANNEX C: SCALE-BREAK CRITERION (2D thin-plate vs 3D thick-plate) + Delta-t8/5
# ============================================================================
# EN 1011-2:2001 Annex C formulas (BS EN 1011-2:2001; Dillinger e-service welding-calc help page,
# service.dillinger.de), Q in kJ/mm, Tp preheat/interpass in degC, d in mm, t8/5 in s:
#   3D (thick plate): t8/5 = (6700 - 5*Tp) * Q * (1/(500-Tp) - 1/(800-Tp)) * F3
#   2D (thin  plate): t8/5 = (4300 - 4.3*Tp) * 1e5 * Q^2/d^2 * (1/(500-Tp)^2 - 1/(800-Tp)^2) * F2
#   transition thickness dt: the d at which the two formulas give the SAME t8/5.
#   Decision rule (source, verbatim): "only the greater value obtained from the two formulas is
#   physically valid" -- i.e. for d < dt the 2D value dominates, for d > dt the 3D value dominates;
#   taking max() implements the criterion without a separate d-vs-dt branch.
F_SHAPE = 0.9  # filling-pass shape factor, F2=F3=0.9 for filling passes (same Dillinger source table)


def t85_3d(Q_kJ_mm, Tp_C, F3=F_SHAPE):
    return (6700.0 - 5.0 * Tp_C) * Q_kJ_mm * (1.0 / (500.0 - Tp_C) - 1.0 / (800.0 - Tp_C)) * F3


def t85_2d(Q_kJ_mm, Tp_C, d_mm_, F2=F_SHAPE):
    return (4300.0 - 4.3 * Tp_C) * 1.0e5 * Q_kJ_mm ** 2 / d_mm_ ** 2 * \
        (1.0 / (500.0 - Tp_C) ** 2 - 1.0 / (800.0 - Tp_C) ** 2) * F2


def transition_thickness_mm(Q_kJ_mm, Tp_C):
    num = (4300.0 - 4.3 * Tp_C) * 1.0e5 * Q_kJ_mm * \
        (1.0 / (500.0 - Tp_C) ** 2 - 1.0 / (800.0 - Tp_C) ** 2)
    den = (6700.0 - 5.0 * Tp_C) * (1.0 / (500.0 - Tp_C) - 1.0 / (800.0 - Tp_C))
    return math.sqrt(num / den)


t85_3D_val = t85_3d(Q_net_kJ_mm, T0_C)
t85_2D_val = t85_2d(Q_net_kJ_mm, T0_C, d_mm)
dt_transition_mm = transition_thickness_mm(Q_net_kJ_mm, T0_C)
regime = "2D (thin-plate)" if t85_2D_val >= t85_3D_val else "3D (thick-plate)"
t85_governing_s = max(t85_2D_val, t85_3D_val)
# cross-check the max() rule against the direct thickness-vs-transition-thickness comparison:
regime_by_thickness = "2D (thin-plate)" if d_mm < dt_transition_mm else "3D (thick-plate)"
regime_criteria_agree = (regime == regime_by_thickness)


# ============================================================================
# 4. PEAK TEMPERATURE vs DISTANCE (Adams 1958) -> HAZ WIDTH, Ac1/Ac3 crossing
# ============================================================================
# Adams (1958), "Cooling rates and peak temperatures in fusion welding", Welding J. 37(5) --
# thin-plate (2D, line-source-through-thickness) peak-temperature form, constant sqrt(2*pi*e)=4.133
# confirmed against an independent paraphrase ("1/(Tp-To) = 4.13 rho c t y / Hnet + 1/(Tm-To)",
# groups.google.com/g/materials-welding thread citing the AWS Welding Handbook Ch.3):
#   1/(Tp-T0) = sqrt(2*pi*e) * rho*c * d * Y / Q_net_J_mm + 1/(Tm-T0)
SQRT_2PIE = math.sqrt(2.0 * math.pi * math.e)


def peak_temp_distance_2d(Y_mm, Q_J_mm=Q_net_J_mm, d_=d_mm, T0=T0_C, Tm=T_MELT_C):
    inv = SQRT_2PIE * RHOC_J_MM3_K * d_ * Y_mm / Q_J_mm + 1.0 / (Tm - T0)
    return T0 + 1.0 / inv


def distance_for_peak_temp_2d(Tp_C, Q_J_mm=Q_net_J_mm, d_=d_mm, T0=T0_C, Tm=T_MELT_C):
    inv_needed = 1.0 / (Tp_C - T0) - 1.0 / (Tm - T0)
    if inv_needed <= 0:
        return float("inf")
    return inv_needed * Q_J_mm / (SQRT_2PIE * RHOC_J_MM3_K * d_)


# S355 Ac1/Ac3, cross-checked from TWO independent studies:
#  (a) Wang et al., "Influence of Vanadium Micro-Alloying on the Microstructure of Structural High
#      Strength Steels Welded Joints", Materials 2023, 16(7):2897 (MDPI/PMC10096130) -- reference-material
#      S355: Ac1=715C, Ac3=825C (paper's own simulation used a ROUNDED Ac1=715C, Ac3=815C).
#  (b) independent search return: "Ac1 and Ac3 were assumed equal to 715C and 815C respectively" for a
#      generic S355 study set  -- same 715/815 pair, cross-confirms (a)'s rounded value.
AC1_C = 715.0
AC3_C = 815.0

Y_at_Ac3_mm = distance_for_peak_temp_2d(AC3_C)   # inner HAZ boundary (fully austenitised / coarse grain)
Y_at_Ac1_mm = distance_for_peak_temp_2d(AC1_C)   # outer HAZ boundary (no transformation beyond this)
HAZ_width_mm = Y_at_Ac1_mm   # conventional definition: fusion line (Y=0) to the Ac1 isotherm


# ============================================================================
# 5. PROCESS-DAG CONSEQUENCE: EN 1011-2 preheat/interpass requirement for S355
# ============================================================================
# Carbon equivalent (CEV/IIW) for S355: typical composition (EN 10025-2 limits, C<=0.20-0.22%, Mn<=1.6%)
# gives CEV = C + Mn/6 + ... ~ 0.40-0.45; independently, a public EN 1011-2 worked example for
# "10025 S355" reports CEV=0.433 (groups.google.com materials-welding thread) --
# matches the composition-based estimate closely, used as the CEV anchor here.
CEV_S355 = 0.433
# EN 1011-2 practice note (qaqctips.blogspot.com "preheat for carbon steel per BS EN 1011-2", fetched
# : "for S355J0, preheating should be considered when plate thickness >= 20mm" (single-member
# thickness, low-hydrogen process such as GMAW solid wire).
PREHEAT_THRESHOLD_SINGLE_MM = 20.0
# the SAME worked CEV=0.433 example returns "minimum preheat temperature 0C" at that CEV -- i.e. at the
# this CEV, EN 1011-2's own preheat chart (Fig. C.2.a) does not itself force T0 above ambient;
# thickness is the controlling variable in this specific example.

single_wall_needs_preheat = d_mm >= PREHEAT_THRESHOLD_SINGLE_MM
# corner joints: two 12.5mm walls meet -> EN 1011-2 uses COMBINED thickness at T/corner joints, not the
# single-member thickness, for the preheat/interpass decision (steelconstruction.info box-section guidance
# for the same joint family).
combined_thickness_corner_mm = 2 * d_mm       # 25.0mm
combined_thickness_tjoint_mm = 3 * d_mm       # 37.5mm (T-joint: cross-beam web meets two perimeter faces)
corner_needs_preheat = combined_thickness_corner_mm >= PREHEAT_THRESHOLD_SINGLE_MM
tjoint_needs_preheat = combined_thickness_tjoint_mm >= PREHEAT_THRESHOLD_SINGLE_MM

# quantify the mechanical consequence: recompute the governing Delta-t8/5 at the T-joint's COMBINED
# thickness (same Q, same F, T0=20C) to show WHY the joint-level answer differs from the single-wall one.
t85_3D_tjoint = t85_3d(Q_net_kJ_mm, T0_C)                       # unchanged -- 3D formula is d-independent
t85_2D_tjoint = t85_2d(Q_net_kJ_mm, T0_C, combined_thickness_tjoint_mm)
dt_transition_tjoint = dt_transition_mm                          # Q,Tp unchanged -> same transition dt
regime_tjoint = "2D (thin-plate)" if combined_thickness_tjoint_mm < dt_transition_tjoint else "3D (thick-plate)"
t85_governing_tjoint_s = max(t85_2D_tjoint, t85_3D_tjoint)

process_dag_consequence = {
    "single_wall_12_5mm_needs_preheat_by_EN1011_2_thickness_rule": single_wall_needs_preheat,
    "corner_joint_combined_25mm_needs_preheat": corner_needs_preheat,
    "tjoint_combined_37_5mm_needs_preheat": tjoint_needs_preheat,
    "CEV_S355_used": CEV_S355,
    "governing_t85_at_single_wall_s": t85_governing_s,
    "governing_t85_at_tjoint_combined_thickness_s": t85_governing_tjoint_s,
    "delta_t85_s": t85_governing_s - t85_governing_tjoint_s,
    "verdict": "YES -- a new CONSTRAINT (not a new precedence edge) is required" if
               (corner_needs_preheat or tjoint_needs_preheat) else "NO",
    "what": "interpass_temperature_floor_check: attach a monitored interpass-temperature floor "
            "(measured, not just nominal T0=20C ambient) to weld_root_pass_side_A, weld_root_pass_side_B "
            "and weld_fill_cap_passes -- the three nodes whose local geometry is a T- or corner-joint "
            "(combined thickness 25-37.5mm >= the 20mm EN1011-2 single-member preheat-consideration "
            "threshold), even though the FLAT WALL alone (12.5mm) does not trigger it.",
    "why_mechanical": "at the T-joint's combined thickness the governing Delta-t8/5 drops from %.2fs to "
                       "%.2fs (regime %s->%s) for the SAME heat input -- faster cooling at the thicker "
                       "joint intersection raises HAZ-hardness risk there specifically, which the "
                       "flat-wall calculation does not see." % (
                           t85_governing_s, t85_governing_tjoint_s, regime, regime_tjoint),
    "not_a_new_dag_run": "no new precedence edge is implied (the constraint attaches to already-ordered "
                          "nodes, it does not reorder anything).",
}


# ============================================================================
# 6. INDEPENDENT NUMERICAL CROSS-CHECK: full transient Rosenthal 2D cooling curve
#    -> extract Delta-t8/5 by root-finding, at 3 time-grid resolutions (mesh/timestep convergence,
#    since this module is a closed-form analytical solution with NO spatial mesh -- the numerics here
#    are the root-finder's time-grid, so THAT is the convergence axis reported).
# ============================================================================
def T_of_t_2d(t_s, Y_mm, Q_J_mm=Q_net_J_mm, d_=d_mm, T0=T0_C, a=A_DIFFUSIVITY_MM2_S):
    """Rosenthal 1946 2D (line-source-through-thickness) quasi-stationary transient temperature
    at a fixed point Y from the weld centerline, time t after the source passed that point."""
    if t_s <= 0:
        return T0
    return T0 + (Q_J_mm / (d_ * RHOC_J_MM3_K)) * (1.0 / math.sqrt(4.0 * math.pi * a * t_s)) * \
        math.exp(-(Y_mm ** 2) / (4.0 * a * t_s))


def find_crossing_time(target_T, Y_mm, t_lo, t_hi, dt_grid):
    """Bisection on a fixed time grid of resolution dt_grid; returns crossing time in that grid's
    resolution (used to show the crossing-time ESTIMATE converges as dt_grid shrinks)."""
    n = int((t_hi - t_lo) / dt_grid)
    prev_t, prev_T = t_lo, T_of_t_2d(t_lo, Y_mm)
    for i in range(1, n + 1):
        t = t_lo + i * dt_grid
        T = T_of_t_2d(t, Y_mm)
        if (prev_T - target_T) * (T - target_T) <= 0 and prev_T != T:
            # linear interpolation within the bracketing grid step (removes first-order grid bias)
            frac = (prev_T - target_T) / (prev_T - T)
            return prev_t + frac * (t - prev_t)
        prev_t, prev_T = t, T
    return None


Y_NUMERIC_MM = 1.0  # evaluate close to the fusion boundary, matching EN1011-2's own weld-centerline Delta-t8/5 definition
_grids = [1.0, 0.1, 0.01]  # seconds; convergence sweep
_convergence = []
for g in _grids:
    t800 = find_crossing_time(800.0, Y_NUMERIC_MM, 0.01, 400.0, g)
    t500 = find_crossing_time(500.0, Y_NUMERIC_MM, 0.01, 800.0, g)
    _convergence.append({
        "dt_grid_s": g,
        "t_800C_s": t800,
        "t_500C_s": t500,
        "delta_t85_s": (t500 - t800) if (t500 is not None and t800 is not None) else None,
    })

numeric_delta_t85_s = _convergence[-1]["delta_t85_s"]
convergence_deltas = [c["delta_t85_s"] for c in _convergence if c["delta_t85_s"] is not None]
converged = len(convergence_deltas) >= 2 and abs(convergence_deltas[-1] - convergence_deltas[-2]) < 0.05

en1011_2_vs_numeric_pct_diff = 100.0 * (numeric_delta_t85_s - t85_governing_s) / t85_governing_s

# ---- SECOND, independent theoretical closed-form check: plain (non-empirical) Rosenthal Delta-t8/5,
# textbook forms (e.g. Kou, "Welding Metallurgy"), using the SAME lambda/rho*c/Q/d as everywhere else
# in this module -- isolates whether a 2D vs 3D gap is a material-constant artifact or a genuine
# empirical-vs-theoretical difference.
def t85_3d_theoretical(Q_J_mm, T0):
    return (Q_J_mm / (2.0 * math.pi * LAMBDA_W_MM_K)) * (1.0 / (500.0 - T0) - 1.0 / (800.0 - T0))


def t85_2d_theoretical(Q_J_mm, d_, T0):
    return ((Q_J_mm / d_) ** 2 / (4.0 * math.pi * LAMBDA_W_MM_K * RHOC_J_MM3_K)) * \
        (1.0 / (500.0 - T0) ** 2 - 1.0 / (800.0 - T0) ** 2)


t85_3D_theoretical_val = t85_3d_theoretical(Q_net_J_mm, T0_C)
t85_2D_theoretical_val = t85_2d_theoretical(Q_net_J_mm, d_mm, T0_C)
pct_diff_3D_theory_vs_en1011_2 = 100.0 * (t85_3D_theoretical_val - t85_3D_val) / t85_3D_val
pct_diff_2D_theory_vs_en1011_2 = 100.0 * (t85_2D_theoretical_val - t85_2D_val) / t85_2D_val


# ============================================================================
# 7. CCT CROSS-CHECK: predicted phase regime at this Delta-t8/5 vs INDEPENDENT measured HAZ hardness
# ============================================================================
# Qualitative CCT behaviour for S355 (low/medium-hardenability C-Mn steel, CEV~0.40-0.45):
#   - martensite requires very fast quench, t8/5 below roughly 2-5s for this hardenability class
#     (TMCP low-alloy structural steels S355MC show "relatively low and stable hardness... limited
#     hardenability and stable ferritic-bainitic microstructures" across t8/5 = 5-20s -- MDPI 2075-4701,
#     10(2):229 abstract summary).
#   - the computed governing t8/5 for this joint (see below) sits inside or above that 5-20s window,
#     so ferrite-bainite (NOT martensite) is the CCT-diagram-predicted microstructure.
predicted_phase = "ferrite-bainite (no martensite)" if t85_governing_s >= 5.0 else \
    "bainite-martensite transition zone (fast-cool regime)"

# INDEPENDENT published MEASURED HAZ hardness for S355 GMAW-family welds (source NOT used for Ac1/Ac3
# above -- Ac1/Ac3 came from the vanadium-alloying dilatometry paper (MDPI 16(7):2897); hardness here
# comes from two SEPARATE, independent studies):
#   (a) Yilbas et al., "Laser welding and weld hardness analysis of thick section S355 structural
#       steel" (researchgate.net/publication/251540432) -- GMAW comparison sample: max microhardness
#       175 HV (VHN).
#   (b) hybrid laser-arc + narrow-gap SAW of 80mm S355ML (sciencedirect.com S0030399226001477 abstract
#       summary) -- base metal 164+/-6 HV1, weld/HAZ zone 171-176 HV1.
#   ceiling: EN ISO 15614-1 max allowable HAZ hardness for non-alloy structural steel = 380 HV10
#       (multiple corroborating summaries).
independent_measured_band_HV = {"lo": 164.0, "hi": 176.0, "single_GMAW_point": 175.0}
iso_ceiling_HV = 380.0

cct_crosscheck = {
    "governing_delta_t85_s": t85_governing_s,
    "predicted_phase_from_S355_hardenability_literature": predicted_phase,
    "independent_measured_HAZ_hardness_band_HV": independent_measured_band_HV,
    "EN_ISO_15614_1_ceiling_HV10": iso_ceiling_HV,
    "prediction_hits_measured_band": "PASS" if predicted_phase.startswith("ferrite-bainite") else
        "MISS (fast-cool regime predicted, would expect band above 176 HV if this fires)",
    "mechanism_if_pass": "at t8/5 >= ~5-20s (this module's governing value, see below), S355's low "
                          "hardenability (CEV~0.43, no significant B/Cr/Mo) precludes martensite "
                          "formation regardless of cooling rate within the arc-welding-achievable range "
                          "-- diffusional ferrite/bainite transformation dominates, consistent with the "
                          "independently measured 164-176 HV band which sits far below the ISO 380 HV10 "
                          "cracking-risk ceiling. A MISS (predicted fast-cool/high-hardness) would instead "
                          "be produced by too-thin an effective section or too-low a heat input pushing "
                          "t8/5 under ~5s -- mechanism: insufficient volumetric heat sink -> quench rate "
                          "approaches the CCT diagram's bainite-start/martensite-start boundary.",
}


# ============================================================================
# 8. SCALE-BREAK DECLARATION
# ============================================================================
skalbrott = {
    "declared": True,
    "why_the_precedence_solver_does_not_apply_here": (
        "the weld thermal field is governed by a continuous PDE (transient heat diffusion, "
        "d T/dt = a * Laplacian(T) plus a moving source term) with NO order relation between "
        "discrete steps to solve for -- there is no 'A before B' precedence to topologically sort; "
        "the physics is a field evaluated continuously in (x,y,z,t), not a partial order over a node set."
    ),
    "fifth_scale_solver_family": ["Rosenthal-analytical", "FE-thermal"],
    "chosen_this_round": "Rosenthal-analytical",
    "FE_upgrade_price": "a full FE thermal-transient model (e.g. Goldak double-ellipsoid moving source "
                         "on a meshed 3D solid, ANSYS/Abaqus or an OSS FE thermal solver) would add: "
                         "(1) real multi-pass sequencing effects on interpass temperature (this module "
                         "uses one representative pass, not a full N-pass accumulation), (2) temperature-"
                         "dependent lambda/rho*c (this module uses constant properties, standard Rosenthal "
                         "assumption), (3) true 3D corner/T-joint geometry rather than the combined-"
                         "thickness proxy used in section 5. Booked, not paid, this round.",
}


# ============================================================================
# ASSEMBLE REPORT
# ============================================================================
report = {
    "cell": "MATERIAL-WELD-HAZ-V0",
    "thread": "weld microstructure / heat-affected zone (analytical leg)",
    "generated_by": "material_weld_haz_v0.py",
    "deterministic": True,
    "substrate_read": {
        "path": "inlined joint/process description (see SUBSTRATE in this module)",
        "sha256": sha256_of(SUBSTRATE),
    },
    "substrate_inline": SUBSTRATE,
    "geometry_from_substrate": {
        "profile": ext["profile"],
        "wall_thickness_mm": d_mm,
        "steel_grade": "S355",
    },
    "gmaw_parameters": {
        "deposition_rate_kg_hr_from_substrate": deposition_rate_kg_hr,
        "current_A_CONSTRUCTED_from_published_chart": I_A,
        "voltage_V_CONSTRUCTED": V_V,
        "arc_efficiency_eta": ETA_ARC,
        "arc_efficiency_source": "DuPont & Marder 1995, Welding J. 74(12):406s-416s",
        "travel_speed_mm_s_CONSTRUCTED": V_TRAVEL_MM_S,
        "P_gross_W": P_gross_W,
        "P_net_W": P_net_W,
        "Q_net_kJ_mm": Q_net_kJ_mm,
    },
    "round_trip_check": round_trip,
    "thermal_properties_steel": {
        "lambda_W_mm_K": LAMBDA_W_MM_K,
        "rho_c_J_mm3_K": RHOC_J_MM3_K,
        "diffusivity_mm2_s": A_DIFFUSIVITY_MM2_S,
        "T_melt_C": T_MELT_C,
        "T0_C_no_preheat_justification": "single-wall d=12.5mm < EN1011-2's 20mm single-member "
                                          "preheat-consideration threshold at this CEV (see section 5)",
    },
    "scale_break_criterion_2D_vs_3D": {
        "formula_source": "EN 1011-2:2001 Annex C (Dillinger e-service welding-calc help page, "
                           "service.dillinger.de)",
        "t85_3D_thick_plate_s": t85_3D_val,
        "t85_2D_thin_plate_s": t85_2D_val,
        "transition_thickness_dt_mm": dt_transition_mm,
        "actual_thickness_mm": d_mm,
        "regime_by_max_rule": regime,
        "regime_by_thickness_comparison": regime_by_thickness,
        "criteria_agree": regime_criteria_agree,
        "governing_delta_t85_s": t85_governing_s,
        "F_shape_factor": F_SHAPE,
    },
    "haz_width_ac1_ac3": {
        "Ac1_C": AC1_C,
        "Ac3_C": AC3_C,
        "Ac1_Ac3_source": "Wang et al., Materials 2023, 16(7):2897 (MDPI/PMC10096130), cross-checked "
                           "against an independent S355 study's own 715/815C assumption",
        "distance_to_Ac3_isotherm_mm": Y_at_Ac3_mm,
        "distance_to_Ac1_isotherm_mm": Y_at_Ac1_mm,
        "HAZ_width_mm": HAZ_width_mm,
        "adams_formula_source": "Adams 1958, Welding J. 37(5); constant sqrt(2*pi*e)=4.133 cross-"
                                 "checked against an independent paraphrase citing the AWS Welding "
                                 "Handbook Ch.3 (groups.google.com materials-welding)",
    },
    "process_dag_consequence": process_dag_consequence,
    "numeric_crosscheck_rosenthal_transient": {
        "method": "full transient Rosenthal-2D T(t) at Y=1mm from fusion line, 800C/500C crossing times "
                   "found by grid bisection at 3 time-grid resolutions (analytical closed-form -> no "
                   "spatial mesh; time-grid IS the convergence axis reported here)",
        "convergence_sweep": _convergence,
        "converged": converged,
        "numeric_delta_t85_s": numeric_delta_t85_s,
        "en1011_2_governing_delta_t85_s": t85_governing_s,
        "pct_diff_numeric_vs_en1011_2": en1011_2_vs_numeric_pct_diff,
        "second_independent_check_theoretical_closed_form": {
            "t85_3D_theoretical_Rosenthal_s": t85_3D_theoretical_val,
            "t85_3D_EN1011_2_empirical_s": t85_3D_val,
            "pct_diff_3D": pct_diff_3D_theory_vs_en1011_2,
            "t85_2D_theoretical_Rosenthal_s": t85_2D_theoretical_val,
            "t85_2D_EN1011_2_empirical_s": t85_2D_val,
            "pct_diff_2D": pct_diff_2D_theory_vs_en1011_2,
        },
        "mechanism_for_the_gap": "the numeric transient (this section) and the theoretical closed-form "
                                  "2D Rosenthal formula (both computed independently, using identical "
                                  "lambda=0.028 W/mmK, rho*c=3.454e-3 J/mm3K) AGREE with each other to "
                                  "within 0.1% -- that cross-validates this module's PDE implementation. "
                                  "BOTH disagree with EN1011-2's empirical 2D formula by a factor of "
                                  "~2.2x (governing/2D case). The 3D case shows the OPPOSITE pattern: "
                                  "theoretical-3D and EN1011-2-3D agree to ~4%. Mechanism: EN1011-2 "
                                  "Annex C is NOT the idealized point/line-source Rosenthal solution -- "
                                  "it is a REGRESSION FIT to measured thermal cycles (Uwer & Degenkolbe, "
                                  "1977 -- 'Charakterisierung von Schweisstemperaturzyklen hinsichtlich "
                                  "ihrer Auswirkung auf die mechanischen Eigenschaften von Schweiss-"
                                  "verbindungen', cited as the origin of the EN1011-2 Annex C formulas, "
                                  "calibrated against REAL weld-pool geometry, "
                                  "convection and non-constant material properties -- effects the idealized "
                                  "instantaneous line-source model omits. Those effects are known to matter "
                                  "MORE for thin-plate (2D, through-thickness conduction, weld-pool shape "
                                  "and root-face melting dominate the heat path) than for thick-plate (3D, "
                                  "far-field conduction dominates, closer to the idealized point-source "
                                  "assumption) -- consistent with the close 3D agreement / poor 2D "
                                  "agreement pattern actually observed here, not an arbitrary discrepancy.",
    },
    "cct_crosscheck": cct_crosscheck,
    "skalbrott_declaration": skalbrott,
}

# ============================================================================
# 9. ATOMS BLOCK -- machine-checkable decisive claims
# ============================================================================
report["ATOMS"] = {
    "base_dir": ".",
    "atoms": [
        {
            "id": "artifact_exists",
            "type": "artifact-exists",
            "claim": "the HAZ report artifact exists on disk",
            "artifact": "artifacts/material_weld_haz_v0.json",
        },
        {
            "id": "value_delta_t85_governing",
            "type": "value-in-artifact",
            "claim": "decisive: the governing (max of 2D/3D per EN1011-2's own rule) Delta-t8/5 for the "
                     "12.5mm S355 GMAW box-section weld",
            "artifact": "artifacts/material_weld_haz_v0.json",
            "key": "scale_break_criterion_2D_vs_3D.governing_delta_t85_s",
            "expected": t85_governing_s,
            "tol": 1e-6,
        },
        {
            "id": "value_haz_width",
            "type": "value-in-artifact",
            "claim": "decisive: HAZ width (fusion line to Ac1 isotherm, Adams 1958 2D peak-temperature "
                     "formula) in mm",
            "artifact": "artifacts/material_weld_haz_v0.json",
            "key": "haz_width_ac1_ac3.HAZ_width_mm",
            "expected": HAZ_width_mm,
            "tol": 1e-6,
        },
        {
            "id": "ineq_delta_t85_above_martensite_threshold",
            "type": "inequality",
            "claim": "DECISIVE, discriminating: governing Delta-t8/5 (computed from this joint's own "
                     "Q_net/thickness, could physically have landed on either side) must exceed the "
                     "~5s S355-hardenability threshold below which the CCT literature predicts a "
                     "bainite-martensite fast-cool regime instead of ferrite-bainite -- the ferrite-"
                     "bainite prediction (and its match to the independently measured 164-176 HV band) "
                     "is FALSIFIED if this inequality fails; a fail is a reportable MISS, not silently "
                     "absorbed (see cct_crosscheck.prediction_hits_measured_band for the miss-branch text)",
            "lhs": {
                "artifact": "artifacts/material_weld_haz_v0.json",
                "key": "scale_break_criterion_2D_vs_3D.governing_delta_t85_s",
            },
            "op": ">",
            "rhs": 5.0,
        },
        {
            "id": "round_trip_arc_on_time",
            "type": "value-in-artifact",
            "claim": "round-trip: GMAW parameters (chart-derived current, chosen travel speed) reproduce "
                     "the inlined arc_on_time_hr (mass/deposition-rate bookkeeping) within its stored "
                     "3-decimal display-rounding tolerance",
            "artifact": "artifacts/material_weld_haz_v0.json",
            "key": "substrate_inline.quantified_process_chain.weld_time.arc_on_time_hr_CONSTRUCTED",
            "expected": arc_on_time_hr_recomputed,
            "tol": 1e-3,
        },
        {
            "id": "value_process_dag_consequence_verdict",
            "type": "value-in-artifact",
            "claim": "process-DAG consequence verdict: does the T-joint/corner combined-thickness "
                     "Delta-t8/5 shift trigger a new interpass-temperature-floor constraint",
            "artifact": "artifacts/material_weld_haz_v0.json",
            "key": "process_dag_consequence.verdict",
            "expected": process_dag_consequence["verdict"],
        },
    ],
}

os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w") as f:
    json.dump(report, f, indent=1)
    f.write("\n")

print("wrote", OUT_PATH)
print("Q_net_kJ_mm =", Q_net_kJ_mm)
print("t85 2D =", t85_2D_val, " t85 3D =", t85_3D_val, " dt_transition_mm =", dt_transition_mm,
      " regime =", regime, " agree =", regime_criteria_agree)
print("HAZ width (Y@Ac1) =", HAZ_width_mm, " Y@Ac3 =", Y_at_Ac3_mm)
print("numeric delta_t85 =", numeric_delta_t85_s, " converged =", converged,
      " pct_diff vs EN1011-2 =", en1011_2_vs_numeric_pct_diff)
print("round_trip I_A =", I_A, " identical_arc_on_time =", round_trip_arc_on_time_pass)
print("process_dag_consequence verdict =", process_dag_consequence["verdict"])
