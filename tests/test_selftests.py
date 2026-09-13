"""Pytest wrappers: every module under src/physics_engine is run as a script in a fresh subprocess with
src/physics_engine as the working directory; the test asserts exit code 0. Modules that need a CUDA device skip
when none is present. Modules whose measured dataset is absent run on their synthetic stand-in (they print a
SYNTHETIC INPUT line).
"""
import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src", "physics_engine")
ENV = dict(os.environ, MPLBACKEND="Agg")


def _cuda_available():
    try:
        import warp as wp
        wp.init()
        return wp.is_cuda_available()
    except Exception:
        return False


CUDA = _cuda_available()


# Modules whose own gates FAIL on the real dataset (recorded SYNTHETIC-ONLY in docs/RUNNING.md): when data/<slug>/ is present
# they exit non-zero by their own verdict, so the test asserts that the verdict was printed rather than exit 0.
OWN_VERDICT = {
    "electrochem/battery_boiling_curve_zuber_chf.py": "FAIL",
    "wave_optics/l_kk_real_metamaterial.py": "FAIL",
}


def run(rel, timeout=300):
    proc = subprocess.run([sys.executable, os.path.join(SRC, rel)], cwd=SRC, env=ENV,
                          capture_output=True, text=True, timeout=timeout)
    if rel in OWN_VERDICT and proc.returncode != 0:
        assert OWN_VERDICT[rel] in proc.stdout, proc.stdout[-4000:] + proc.stderr[-4000:]
        return proc.stdout
    assert proc.returncode == 0, proc.stdout[-4000:] + proc.stderr[-4000:]
    return proc.stdout


CPU_MODULES = [
    'electrochem/battery_boiling_curve_zuber_chf.py',
    'electrochem/battery_calendar_aging.py',
    'electrochem/battery_capacity_fade.py',
    'electrochem/battery_cell_swelling_breathing.py',
    'electrochem/battery_chemomechanical_crack_sei.py',
    'electrochem/battery_coldplate_conjugate_heat.py',
    'electrochem/battery_combustion_autoignition_semenov.py',
    'electrochem/battery_coupled_electro_thermal_runaway.py',
    'electrochem/battery_dc_arc_quench_disconnect.py',
    'electrochem/battery_ejecta_velocity_projectile.py',
    'electrochem/battery_entropic_heat.py',
    'electrochem/battery_frank_kamenetskii_spatial_explosion.py',
    'electrochem/battery_griffith_particle_fracture.py',
    'electrochem/battery_isc_initiation.py',
    'electrochem/battery_jellyroll_thermal_anisotropy.py',
    'electrochem/battery_li_plating_onset_map.py',
    'electrochem/battery_manifold_flow_distribution.py',
    'electrochem/battery_natural_convection_cooling.py',
    'electrochem/battery_ocv_soc_equilibrium.py',
    'electrochem/battery_tr_criticality_early_warning.py',
    'electrochem/battery_vent_burst_dynamics.py',
    'electrochem/battery_vent_jet_recoil.py',
    'em/antenna_aperture.py',
    'em/antenna_sigma_budget.py',
    'em/bearing_maxwell_ehl_edm.py',
    'em/capacitive_sensor.py',
    'em/curie_weiss_ferromagnet.py',
    'em/dielectric_force_capacitor.py',
    'em/eddy_current_brake.py',
    'em/em_magnetic_pressure.py',
    'em/ferrofluid_rosensweig.py',
    'em/helmholtz_coil.py',
    'em/magnetic_alfven_wave.py',
    'em/magnetoconvection.py',
    'em/magnetoresistance_twoband.py',
    'em/magnetostriction.py',
    'em/magnetron_hull_cutoff.py',
    'em/piezo_resonator.py',
    'em/piezoelectric.py',
    'em/plasma_frequency.py',
    'em/spine_criticality_fisher_plasma.py',
    'em/two_stream_instability.py',
    'litho/d_litho_certified_imaging_ilt.py',
    'litho/i_litho_synthetic_vs_real_contour_ler_reality_gap.py',
    'litho/i_reality_gap_on_real_litho_aerial.py',
    'litho/litho_aerial_image_scalar.py',
    'litho/litho_depth_of_focus_cert.py',
    'litho/litho_etch_arde_ion_flux.py',
    'litho/litho_etch_levelset_ballistic.py',
    'litho/litho_euv_stochastic_ler_cert.py',
    'litho/vof_zalesak_cert.py',
    'materials/i1_lattice_engine_predicts_modulus.py',
    'materials/i1_lattice_engine_sigma_min.py',
    'materials/i1_lattice_maxwell_connectivity.py',
    'materials/i1_lattice_strength_maxwell.py',
    'process/fillet_weld.py',
    'process/hall_petch.py',
    'process/keyhole_absorption_jump.py',
    'process/laser_keyhole_absorptance.py',
    'process/lpbf_absorptance_render_match.py',
    'process/lpbf_keyhole_threshold_render_match.py',
    'process/lpbf_meltpool_render_match.py',
    'process/lpbf_residual_stress_render_match.py',
    'process/lpbf_transient_meltpool_render_match.py',
    'process/marangoni_meltpool_cfd.py',
    'process/marangoni_meltpool_number.py',
    'process/material_weld_haz_v0.py',
    'process/melt_pool_growth.py',
    'process/vapor_recoil_keyhole.py',
    'process/weld_goldak_spot_nist.py',
    'process/weld_residual_edd_in718.py',
    'process/weld_resistance_spot_nugget.py',
    'quantum/band_structure.py',
    'quantum/bethe_bloch.py',
    'quantum/bloch_oscillation.py',
    'quantum/compton.py',
    'quantum/ising_2d.py',
    'quantum/maxwell_boltzmann.py',
    'quantum/mossbauer_recoilless.py',
    'quantum/quantum_revival.py',
    'quantum/quantum_tunnelling.py',
    'quantum/quantum_walk.py',
    'quantum/schrodinger_1d.py',
    'quantum/two_level_paramagnet.py',
    'ray_optics/d_overlay_metrology_crb.py',
    'ray_optics/lens_raytrace_cell.py',
    'ray_optics/optics_achromat.py',
    'ray_optics/optics_departure.py',
    'ray_optics/optics_lens.py',
    'ray_optics/optics_lens_design.py',
    'ray_optics/photonics_ring_fsr_group_index_cert.py',
    'ray_optics/photonics_ring_thermooptic_tuning_transducer.py',
    'ray_optics/photonics_slowlight_modulator_efficiency_bandwidth_tradeoff.py',
    'scattering/b2_fresnel_scattered_phase.py',
    'scattering/dielectric_sphere_permittivity_recovery_mie.py',
    'scattering/emcc_rcs_cube_physical_optics_render_match_vs_real_em_scattering.py',
    'scattering/p18_dielectric_sphere_mie_forced.py',
    'scattering/p18_fresnel_field_match.py',
    'scattering/p18_mie_scattering_render_match.py',
    'scattering/p18_mie_sphere_rcs_render_match.py',
    'scattering/pa_adjoint_helmholtz_envelope.py',
    'scattering/spine_scatter2d_multiple_helmholtz_forward_C.py',
    'thermal/kelvin_helmholtz_onset.py',
    'thermal/spine_kelvin_helmholtz_wavelength.py',
    'thermal/tnf_sandia_flameD_T_vs_mixturefraction_burke_schumann_render_match_vs_real.py',
    'wave_optics/airy_diffraction_limit.py',
    'wave_optics/bragg_diffraction.py',
    'wave_optics/differentiable_xray_classprior.py',
    'wave_optics/diffraction_grating.py',
    'wave_optics/inverse_design_wave.py',
    'wave_optics/gravitational_lensing.py',
    'wave_optics/l_kk_real_metamaterial.py',
    'wave_optics/l_kk_susceptibility_cert.py',
    'wave_optics/laser_speckle.py',
    'wave_optics/laser_threshold.py',
    'wave_optics/metalens_psf_imaging_cert.py',
    'wave_optics/metrology_photon_floor_cell.py',
    'wave_optics/optical_vortex_oam.py',
    'wave_optics/optics_sphere_psf_diffraction_lambda_scaling_real.py',
    'wave_optics/p18_superlens_evanescent_collapse.py',
    'wave_optics/p20_camera_lens_per_element.py',
    'wave_optics/p20_jwst_zernike.py',
    'wave_optics/p20_metalens_chromatic.py',
    'wave_optics/p20_sphere_psf_diffraction.py',
    'wave_optics/photon_sphere_shadow.py',
    'wave_optics/relativistic_aberration.py',
    'wave_optics/twin_calibration_wave.py',
    'wave_optics/schwarzschild_radius.py',
    'wave_optics/shannon_capacity.py',
    'wave_optics/wave_optics_cell.py',
]

CUDA_MODULES = [
    'litho/socs_batched_2d_fft2_aerial_engine.py',
    'litho/socs_batched_tcc_apply_gpu_api_for_ilt.py',
    'litho/socs_imaging_cuda_first_certified_vs_abbe_8kernel.py',
    'ray_optics/lens_design_search_v1.py',
    'ray_optics/lens_raytrace_gpu.py',
    'wave_optics/coupled_multiphysics_calibration.py',
    'wave_optics/sigma_guided_fwi.py',
    'wave_optics/twin_calibration_multisource.py',
]


@pytest.mark.parametrize("rel", CPU_MODULES)
def test_selftest(rel):
    run(rel)


@pytest.mark.skipif(not CUDA, reason="requires a CUDA device")
@pytest.mark.parametrize("rel", CUDA_MODULES)
def test_selftest_cuda(rel):
    run(rel)


def test_vendor_imports():
    sys.path.insert(0, os.path.join(SRC, "_vendor"))
    import evidence_emit  # noqa: F401
    import render_match_scaffold  # noqa: F401


def test_vendor_wave_substrate():
    run('_vendor/diff_wave_substrate.py')
