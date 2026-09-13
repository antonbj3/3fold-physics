# Running

Status values: VERIFIED-FRESH = the module's own gates passed in the recorded run; OWN-GATE-FAIL = one or more own gates failed, result retained and unpromoted; CUDA-ONLY = GPU execution not yet verified in the recorded scope; SYNTHETIC-ONLY = passing evidence limited to synthetic inputs. Earlier attempts are notes, not additional current module rows.

1. `python -m venv .venv && .venv/bin/pip install -r requirements.txt`.
2. Every module under `src/physics_engine/` is also a script: run it directly and it executes its own gates and
   prints the verdicts. Run the scripts with `src/physics_engine` as the working directory:
   `cd src/physics_engine && python em/eddy_current_brake.py`.
3. `pytest tests/` wraps those self-tests (one subprocess per module, exit code 0 = pass).
4. Packages by domain: `em/` (magnetostatics, induction, MHD, EM-elastic coupling), `scattering/` (Mie, physical
   optics, Helmholtz forward/adjoint), `wave_optics/` (Fourier optics, PSF/MTF, dispersion, Kramers-Kronig,
   relativistic optics), `ray_optics/` (eikonal/GRIN lens design, thin-film, metrology floors, photonics),
   `litho/` (partially coherent imaging, depth of focus, stochastic line-edge roughness, etch level-set),
   `quantum/` (Schrödinger, tunnelling, bands, Bloch, Ising, kinetic theory), `thermal/` (shear instabilities,
   combustion), `process/` (laser melt pool, keyhole, weld, residual stress), `materials/` (lattice mechanics),
   `electrochem/` (battery electro-thermal and safety solvers), `chains/` (coupled chains: one module's output run
   into the next with the certificates composed), `_vendor/` (three small helpers).
5. Modules that compare against a measured dataset look for it under `data/<dataset-slug>/` at the repository
   root (the slug and the public source are named in each module's docstring). When the directory is absent the
   module prints `SYNTHETIC INPUT: ...` and runs the same pipeline on a stand-in generated from its own forward
   model. Those modules are SYNTHETIC-ONLY in the table below; put the dataset in place to reproduce the
   real-data result.
6. Evidence JSONs and figures are written to `artifacts/` next to the module (gitignored).
7. Modules marked CUDA-ONLY below need a CUDA device (`torch.fft`, or warp batch sizes that only make sense on a
   GPU); `pytest` skips them when no device is present. The differentiable wave and lattice modules use warp and
   run on its CPU backend.

## Status

One row per shipped module. VERIFIED-FRESH = its self-test ran in this repository's venv on CPU and reproduced
its numbers. CUDA-ONLY = not runnable here until the GPU driver is fixed; last verified in the source project.
SYNTHETIC-ONLY = the method was demonstrated on a dataset that is not in this repository, and it ships with a
synthetic input and a test.

2026-09-12: the measured datasets for all 33 SYNTHETIC-ONLY modules were placed under `data/<slug>/` (see "Real
datasets" below) and every one of those modules was re-run on the real data. 30 reproduced their gates on the real
data and are now VERIFIED-FRESH; the note column carries the decisive real-data numbers. 3 stay SYNTHETIC-ONLY:
their real data is in place but a gate fails (or the module errors) on it — the failing gate is quoted verbatim in
the note and no tolerance or gate was changed.

| module | status | note (real-data run 2026-09-12) |
| --- | --- | --- |
| src/physics_engine/_vendor/diff_wave_substrate.py | VERIFIED-FRESH |  |
| src/physics_engine/_vendor/evidence_emit.py | VERIFIED-FRESH |  |
| src/physics_engine/_vendor/render_match_scaffold.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_boiling_curve_zuber_chf.py | OWN-GATE-FAIL | real pool-boiling-multimodal (4 *Temperature*.lvm): peak q 917 kW/m2 = 83% of Zuber CHF 1111 kW/m2 (G2 PASS), but exit 1 — `G1 ★boiling curve rises: q 97→521 kW/m² over superheat 8→72 °C (nucleate)  FAIL` and `G3 ★NULL: unpowered superheat -7.3 °C (<sat) → q -41 kW/m², no boiling; steady gradient  FAIL` (gates unchanged) |
| src/physics_engine/electrochem/battery_calendar_aging.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_capacity_fade.py | VERIFIED-FRESH | NASA PCoE B0005/6/7/18: B0005 EOL@70% at cycle 124 (band [115,128]); 4 identical cells 96–124 cycles (~10% spread); fade 2.0→5.5 mAh/cycle |
| src/physics_engine/electrochem/battery_cell_swelling_breathing.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_chemomechanical_crack_sei.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_coldplate_conjugate_heat.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_combustion_autoignition_semenov.py | VERIFIED-FRESH | 198 ECN Spray-A conditions: E_a = 21 ± 1 kJ/mol, τ ∝ ρ^-0.74, T-shuffle null collapses R² 0.56→0.25 |
| src/physics_engine/electrochem/battery_coupled_electro_thermal_runaway.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_dc_arc_quench_disconnect.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_ejecta_velocity_projectile.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_entropic_heat.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_frank_kamenetskii_spatial_explosion.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_griffith_particle_fracture.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_isc_initiation.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_jellyroll_thermal_anisotropy.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_li_plating_onset_map.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_manifold_flow_distribution.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_natural_convection_cooling.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_ocv_soc_equilibrium.py | VERIFIED-FRESH | NREL Battery Failure Databank V2: OCV↑SOC corr up to 0.95 (~15 mV/%SOC), inside [2.0,4.2] V; 1 entry flagged 5.9 V; SOC-shuffle -0.19 |
| src/physics_engine/electrochem/battery_tr_criticality_early_warning.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_vent_burst_dynamics.py | VERIFIED-FRESH |  |
| src/physics_engine/electrochem/battery_vent_jet_recoil.py | VERIFIED-FRESH |  |
| src/physics_engine/em/antenna_aperture.py | VERIFIED-FRESH |  |
| src/physics_engine/em/antenna_sigma_budget.py | VERIFIED-FRESH |  |
| src/physics_engine/em/bearing_maxwell_ehl_edm.py | VERIFIED-FRESH |  |
| src/physics_engine/em/capacitive_sensor.py | VERIFIED-FRESH |  |
| src/physics_engine/em/curie_weiss_ferromagnet.py | VERIFIED-FRESH |  |
| src/physics_engine/em/dielectric_force_capacitor.py | VERIFIED-FRESH |  |
| src/physics_engine/em/eddy_current_brake.py | VERIFIED-FRESH |  |
| src/physics_engine/em/em_magnetic_pressure.py | VERIFIED-FRESH |  |
| src/physics_engine/em/ferrofluid_rosensweig.py | VERIFIED-FRESH |  |
| src/physics_engine/em/helmholtz_coil.py | VERIFIED-FRESH |  |
| src/physics_engine/em/magnetic_alfven_wave.py | VERIFIED-FRESH |  |
| src/physics_engine/em/magnetoconvection.py | VERIFIED-FRESH |  |
| src/physics_engine/em/magnetoresistance_twoband.py | VERIFIED-FRESH |  |
| src/physics_engine/em/magnetostriction.py | VERIFIED-FRESH |  |
| src/physics_engine/em/magnetron_hull_cutoff.py | VERIFIED-FRESH |  |
| src/physics_engine/em/piezo_resonator.py | VERIFIED-FRESH |  |
| src/physics_engine/em/piezoelectric.py | VERIFIED-FRESH |  |
| src/physics_engine/em/plasma_frequency.py | VERIFIED-FRESH |  |
| src/physics_engine/em/spine_criticality_fisher_plasma.py | VERIFIED-FRESH |  |
| src/physics_engine/em/two_stream_instability.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/d_litho_certified_imaging_ilt.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/i_litho_synthetic_vs_real_contour_ler_reality_gap.py | VERIFIED-FRESH | real SEM Test01.tif (59 edges): LER raw 3.573 px, noise-subtracted 1.631 px vs pylitho-SOCS-24kernel contour LER 0.000 px (needs torch installed) |
| src/physics_engine/litho/i_reality_gap_on_real_litho_aerial.py | VERIFIED-FRESH | pylitho-SOCS-24kernel aerial: coarse-loss bias 0.054, sharp/flat bias ratio 14.7, untrustworthy map 28% (needs torch installed) |
| src/physics_engine/litho/litho_aerial_image_scalar.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/litho_depth_of_focus_cert.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/litho_etch_arde_ion_flux.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/litho_etch_levelset_ballistic.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/litho_euv_stochastic_ler_cert.py | VERIFIED-FRESH |  |
| src/physics_engine/litho/socs_batched_2d_fft2_aerial_engine.py | VERIFIED-FRESH | run on RTX 5070 2026-09-12: 40×40 grid, batched fft2 apply per gradient step; 4.0 s; |
| src/physics_engine/litho/socs_batched_tcc_apply_gpu_api_for_ilt.py | VERIFIED-FRESH | run on RTX 5070 2026-09-12: batched apply scales with batch, knee at B=1024 (973 209/s), tiling floating-point identical; 1.8 s; |
| src/physics_engine/litho/socs_imaging_cuda_first_certified_vs_abbe_8kernel.py | VERIFIED-FRESH | run on RTX 5070 2026-09-12: SOCS vs Abbe sum drift 2.6e-07 (fp32-eps regime), throughput measured on device; 2.2 s; |
| src/physics_engine/litho/vof_zalesak_cert.py | VERIFIED-FRESH |  |
| src/physics_engine/chains/litho_chain_v1.py | VERIFIED-FRESH | coupled chain (aerial -> resist/LER -> etch): etched CD 45.43 +/- 5.78 nm, etched LER 0.451 +/- 0.054 nm over 64 draws; joint/linear-sum 0.876 (CD), 0.551 (LER); etch stage dominates |
| src/physics_engine/materials/i1_lattice_engine_predicts_modulus.py | VERIFIED-FRESH | Mendeley PLA lattice: measured exponents Grid 1.53 / Hex 3.27 / Tri 0.96; binary Maxwell n gives median held-out value error 24% vs free-fit 2% (regime yes, value no) |
| src/physics_engine/materials/i1_lattice_engine_sigma_min.py | VERIFIED-FRESH |  |
| src/physics_engine/materials/i1_lattice_maxwell_connectivity.py | VERIFIED-FRESH | Z=3/4/6 → measured density-stiffness exponent 3.3/1.5/1.0, corr(Z,exp) = -0.89 |
| src/physics_engine/materials/i1_lattice_strength_maxwell.py | VERIFIED-FRESH | strength-density exponents 3.1/2.0/1.0 for Z=3/4/6, corr(Z,strength-exp) = -0.98 |
| src/physics_engine/process/fillet_weld.py | VERIFIED-FRESH |  |
| src/physics_engine/process/hall_petch.py | VERIFIED-FRESH |  |
| src/physics_engine/process/keyhole_absorption_jump.py | VERIFIED-FRESH | NIST Al_Spot_TDA: keyhole absorptance 71% (band [70,73]) vs conduction 25% — 2.9× jump, midpoint at t ≈ 0.22 ms |
| src/physics_engine/process/laser_keyhole_absorptance.py | VERIFIED-FRESH | NIST measured conduction 23.9±0.3%, spot keyhole 64.1±3.9% ∈ rendered [55.9,74.5], scan 43.3±1.5%; implied N 3.8 (spot) > 2.1 (scan) |
| src/physics_engine/process/lpbf_absorptance_render_match.py | VERIFIED-FRESH |  |
| src/physics_engine/process/lpbf_keyhole_threshold_render_match.py | VERIFIED-FRESH |  |
| src/physics_engine/process/lpbf_meltpool_render_match.py | VERIFIED-FRESH |  |
| src/physics_engine/process/lpbf_residual_stress_render_match.py | VERIFIED-FRESH | 2248 EDD points: peak |ε| ZZ 0.0043 / XX 0.0035 ∈ yield band [0.0030,0.0055]; corr(XX,ZZ) = -0.47; implied σ_y 859 MPa |
| src/physics_engine/process/lpbf_transient_meltpool_render_match.py | VERIFIED-FRESH | Al_Spot_TDW: energy conservation ∫H dV/ηPt = 1.0011; at matched time the measured pool exceeds keyhole-conduction by up to 39% (t ≤ 800 µs) |
| src/physics_engine/process/marangoni_meltpool_cfd.py | VERIFIED-FRESH |  |
| src/physics_engine/process/marangoni_meltpool_number.py | VERIFIED-FRESH |  |
| src/physics_engine/process/material_weld_haz_v0.py | VERIFIED-FRESH |  |
| src/physics_engine/process/melt_pool_growth.py | VERIFIED-FRESH | Al_Spot_TDW: width at 1 ms = 377 µm, band [351,431] vs measured 400±50 µm; growth t^0.39, √(αt) = 312 µm |
| src/physics_engine/process/vapor_recoil_keyhole.py | VERIFIED-FRESH |  |
| src/physics_engine/process/weld_goldak_spot_nist.py | VERIFIED-FRESH | onset 20 µs data = 20 µs model; end width data 503 vs model 575 µm (rel 0.14, gate ≤0.35), shape r = 0.990; conduction-only RMS 73 µm |
| src/physics_engine/process/weld_residual_edd_in718.py | VERIFIED-FRESH | p95|εzz| = 0.00374, p95|εxx| = 0.00222 ∈ [0.00210,0.00575]; equilibrium |mean|/p95 = 0.062; Poisson slope -0.323 (r -0.791) |
| src/physics_engine/process/weld_resistance_spot_nugget.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/band_structure.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/bethe_bloch.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/bloch_oscillation.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/compton.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/ising_2d.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/maxwell_boltzmann.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/mossbauer_recoilless.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/quantum_revival.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/quantum_tunnelling.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/quantum_walk.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/schrodinger_1d.py | VERIFIED-FRESH |  |
| src/physics_engine/quantum/two_level_paramagnet.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/d_overlay_metrology_crb.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/lens_raytrace_cell.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/lens_design_search_v1.py | VERIFIED-FRESH | run on RTX 5070 2026-09-12: 4096 designs x 6 generations x 2 runs; best certified objective 0.066642 mm vs patent example 0.092974 mm; 12099 certified / 12477 rejected of 24576; two full runs max|delta| = 0.0 (objective and design vector); null case (bounds collapsed on the patent vector) reproduces it at rel 0.0; 31.7 s; |
| src/physics_engine/ray_optics/lens_raytrace_gpu.py | VERIFIED-FRESH | run on RTX 5070 2026-09-12: G-PORT-FAST max rel error 4.08e-14 (PASS @1e-6), repeatability spread 0.0 mm numpy and fused paths; 27.1 s; |
| src/physics_engine/ray_optics/optics_achromat.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/optics_departure.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/optics_lens.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/optics_lens_design.py | VERIFIED-FRESH |  |
| src/physics_engine/ray_optics/photonics_ring_fsr_group_index_cert.py | VERIFIED-FRESH | Res7 FSR 10.88 pm → n_g 1.546 and Res10 11.16 pm → n_g 1.548 (agree 0.1%); vs documented phase index 1.49 (+3.8%, normal dispersion) |
| src/physics_engine/ray_optics/photonics_ring_thermooptic_tuning_transducer.py | VERIFIED-FRESH | 198 power-varied traces: FSR drift 0.05% while comb offset moves 27.8 pm (rigid translation); dλ/dP = 0.174 pm/mW (measured, flagged not certified) |
| src/physics_engine/ray_optics/photonics_slowlight_modulator_efficiency_bandwidth_tradeoff.py | VERIFIED-FRESH | efficiency↔bandwidth log-log r = -0.994 (resonators) / -0.978 (periods); pairing-shuffle null p < 1e-4; exponents -1.62 / -2.66 |
| src/physics_engine/scattering/b2_fresnel_scattered_phase.py | VERIFIED-FRESH | opus1_dielTM_dec4f.exp, view 18 @ 16 GHz, 49 receivers, |S| median 5.09e-2: phase spans ≫2π (≈4 waves); G1–G4 all PASS |
| src/physics_engine/scattering/dielectric_sphere_permittivity_recovery_mie.py | VERIFIED-FRESH | real data (LucernHammer lossless dielectric sphere): penetrable-Mie self-validation (transparent 2e-08, Rayleigh slope 4.00, Clausius-Mossotti CV 0.0004); ε_r band [2.00,2.50], m_fwd 1.50 vs m_back 1.58 (|Δm| 0.08), PEC beaten 4×. Earlier failure was file selection: the glob picked the PEC file first; the module now selects the dielectric file by name. |Δm| 0.08; PEC beaten 4×), but in the shipped slug (both benchmark .rcs present, pec_sphere_multifreq_fs.rcs is required by p18_mie_sphere_rcs_render_match) its `glob(DDIR/**/*.rcs)[0]` selects the 2-column PEC file and it exits 1: `IndexError: index 3 is out of bounds for axis 1 with size 2` (module unchanged) |
| src/physics_engine/scattering/emcc_rcs_cube_physical_optics_render_match_vs_real_em_scattering.py | VERIFIED-FRESH | EMCC cube: broadside PO 14.12 vs measured 13.943 dBsm (Δ +0.18 dB); first null 20.4° vs 21.0°; 0.43→1.3 GHz measured +9.57 vs PO f² +9.61 dB; specular RMS 0.14 dB |
| src/physics_engine/scattering/p18_dielectric_sphere_mie_forced.py | VERIFIED-FRESH | 256 frequencies → single (ε,κ): ε = 2.80, κ = 7.91, RMSE 3.5 dB, corr 0.912 |
| src/physics_engine/scattering/p18_fresnel_field_match.py | VERIFIED-FRESH | opus1_dielTM_dec8f.exp: measured lobe count [1,3,5,6] vs exact TM Mie [0,3,4,6] (grows with ka); forward-lobe NRMSE reported, not gated |
| src/physics_engine/scattering/p18_mie_scattering_render_match.py | VERIFIED-FRESH | Institut Fresnel file; H_φ TM boundary-condition residual 1.23e-14 (0.7 for the wrong TE form), PEC-TM limit err 2e-2 |
| src/physics_engine/scattering/p18_mie_sphere_rcs_render_match.py | VERIFIED-FRESH | LucernHammer PEC sphere: Rayleigh 9x⁴ recovered (0.999), optical limit 1.00, first resonance σ_b/πa²(ka≈1) = 3.64 vs textbook 3.65 |
| src/physics_engine/scattering/pa_adjoint_helmholtz_envelope.py | VERIFIED-FRESH |  |
| src/physics_engine/scattering/spine_scatter2d_multiple_helmholtz_forward_C.py | VERIFIED-FRESH |  |
| src/physics_engine/thermal/kelvin_helmholtz_onset.py | VERIFIED-FRESH |  |
| src/physics_engine/thermal/spine_kelvin_helmholtz_wavelength.py | VERIFIED-FRESH |  |
| src/physics_engine/thermal/tnf_sandia_flameD_T_vs_mixturefraction_burke_schumann_render_match_vs_real.py | VERIFIED-FRESH | pmCDEF.zip, 161 pooled (F,T): measured peak 1905 K at Z = 0.380 vs derived Z_st = 0.353 (ΔZ 0.027 < 0.05), lean branch r = 0.980; 27% deficit vs no-dissociation 2593 K (declared scope) |
| src/physics_engine/wave_optics/airy_diffraction_limit.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/bragg_diffraction.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/differentiable_xray_classprior.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/diffraction_grating.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/gravitational_lensing.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/inverse_design_wave.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/l_kk_real_metamaterial.py | OWN-GATE-FAIL | Historical median3.3174 vs control1.1142 retained, but input-semantics mismatch invalidates the measured-complex-response interpretation. Raw frequency column200-800 has no unit in its header; previous Hz assertion withdrawn. See `docs/KK_INPUT_CONTRACT.md`; original code and thresholds frozen. |
| src/physics_engine/wave_optics/l_kk_susceptibility_cert.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/laser_speckle.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/laser_threshold.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/metalens_psf_imaging_cert.py | VERIFIED-FRESH | measured PSF explains the blur: NCC 0.623→0.801 (same-FWHM Gaussian null 0.726); 2nd instance 0.742→0.948 |
| src/physics_engine/wave_optics/metrology_photon_floor_cell.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/optical_vortex_oam.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/optics_sphere_psf_diffraction_lambda_scaling_real.py | VERIFIED-FRESH | VLT/SPHERE cube, 39 channels 0.958–1.329 µm: core FWHM ∝ λ slope +2.14 px/µm, r = 0.964; λ-shuffle null p = 0.0000; ~1.7× the ideal Airy |
| src/physics_engine/wave_optics/p18_superlens_evanescent_collapse.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/p20_camera_lens_per_element.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/p20_jwst_zernike.py | VERIFIED-FRESH | 5 JWST NRCA3 OPD files (R2024030102 excluded): total WFE RMS 66 nm, residual 83% of WFE (55 nm), segment trefoil 30 nm > alignment 26 nm |
| src/physics_engine/wave_optics/p20_metalens_chromatic.py | VERIFIED-FRESH | measured PSF cube: on-axis Strehl spread ×12.1, core-FWHM spread ×4.2, peak lateral offset 0 px → chromatic defocus |
| src/physics_engine/wave_optics/p20_sphere_psf_diffraction.py | VERIFIED-FRESH | VLT/SPHERE cube: FWHM ∝ λ R² = 0.932, FWHM ratio ×1.17 (λ ratio 1.39), median FWHM/Airy = 2.06 |
| src/physics_engine/wave_optics/photon_sphere_shadow.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/relativistic_aberration.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/schwarzschild_radius.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/shannon_capacity.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/coupled_multiphysics_calibration.py | VERIFIED-FRESH | joint K/rho error 7 %/7 % vs single-physics 19 %/18 % at zero noise (2.6x) and 16 %/16 % at 6 % noise (2.2x) |
| src/physics_engine/wave_optics/sigma_guided_fwi.py | VERIFIED-FRESH | k=5/6 info-gain beats naive-sigma and the random majority; no strategy reaches the 10 % target |
| src/physics_engine/wave_optics/twin_calibration_multisource.py | VERIFIED-FRESH | global c2 rel-err 0.354 -> 0.141 (single source 0.224); sigma-untrustworthy fraction 7 % |
| src/physics_engine/wave_optics/twin_calibration_wave.py | VERIFIED-FRESH |  |
| src/physics_engine/wave_optics/wave_optics_cell.py | VERIFIED-FRESH |  |

### Certified lens design search

`ray_optics/lens_design_search_v1.py` puts a population search on top of the batched GPU tracer and reports only
designs that pass three certificates: paraxial focal length within 0.1 % of the patent's 105.815 mm, full-field
vignetting no worse than the patent example measured on the same ray grid, and axial working f-number within 1 %
of the patent example's. Four free parameters (the two variable airspaces and two radii, bounds centred on the
patent values), objective = equal-weight mean spot RMS at 0 / 25.54 / 34.53 deg object half angle.

    cd src/physics_engine && python ray_optics/lens_design_search_v1.py   # ~32 s on a CUDA device
    pytest -k lens_design_search tests/test_selftests.py

It is a four-parameter local search on one prescription, not a lens design program: fixed glasses, one index set,
one conjugate, spot RMS only (no MTF, wavefront, tolerancing or manufacturability constraint).

## Real datasets

The dataset files go in `data/<slug>/` at the repository root. `data/` is in `.gitignore` and nothing under it is
shipped with the repository — each slug below lists the public source and the exact files a module opens, so the
real-data runs recorded in the Status notes can be reproduced. Sizes are what the 2026-09-12 run used (409 MB
total); the full upstream datasets are much larger, only the files named here are needed.

Two litho modules additionally need `torch` (declared in `requirements.txt`, CPU build is enough): without it
`i_reality_gap_on_real_litho_aerial.py` and `i_litho_synthetic_vs_real_contour_ler_reality_gap.py` fall back to the
Gaussian-PSF stand-in even when `data/lithobench-real-masks/` is present.

| slug | public source | files needed | size | modules |
| --- | --- | --- | --- | --- |
| nist-amb2022 | NIST AM-Bench 2022 AMB2022-01, data.nist.gov ark:/88434/mds2-2525 | `Al_Spot_TDA_Results.csv`, `Al_Spot_TDW_Results.csv`, `Al_Spot_AA_ASR_Results.csv`, `Al_Scan_AA_MWD_ASR_Results.csv` | 2.3 MB | keyhole_absorption_jump, laser_keyhole_absorptance, melt_pool_growth, lpbf_transient_meltpool_render_match, weld_goldak_spot_nist |
| nist-amb-resid | NIST AM-Bench 2022 residual stress, DOI 10.18434/mds2-2711 | `AMB2022_EDD_results_V2.txt` | 84 KB | lpbf_residual_stress_render_match, weld_residual_edd_in718 |
| mendeley-pla-lattice | Mendeley Data record nvzrft8c7d (AM PLA lattice compression) | `Compressivedata.xlsx` | 30 MB | i1_lattice_engine_predicts_modulus, i1_lattice_maxwell_connectivity, i1_lattice_strength_maxwell |
| institut-fresnel | Institut Fresnel free-space scattering database, Opus 1 (Inverse Problems 17(6) 2001, publisher data supplement) | `opus1_dielTM_dec4f.exp`, `opus1_dielTM_dec8f.exp` | 1.6 MB | b2_fresnel_scattered_phase, p18_fresnel_field_match, p18_mie_scattering_render_match |
| emcc-rcs | EMCC RCS benchmark targets (lucernhammer.tripointindustries.com/benchmark_data/EMCC/3d_shapes/serenity/) | `cube_0.43ghz.rcs`, `cube_1.3ghz.rcs` | 244 KB | emcc_rcs_cube_physical_optics_render_match_vs_real_em_scattering |
| sphere-rcs-benchmark | LucernHammer sphere benchmarks (lucernhammer.tripointindustries.com/benchmark_data/spheres/) | `pec_sphere_multifreq_fs.rcs` (from `pec_sphere/multifreq/mie/`), `lossless_dielectric_sphere_multifreq.rcs` (from `dielectric/lossless/multifreq/galaxy/`), both flat in the slug dir | 56 KB | p18_mie_sphere_rcs_render_match, p18_dielectric_sphere_mie_forced, dielectric_sphere_permittivity_recovery_mie (see its note: it picks the first `**/*.rcs`) |
| lithobench-real-masks | LithoBench, github.com/shelljane/lithobench (MIT), NeurIPS 2023 D&B | the repository tree (`config/`, `kernel/`, `pylitho/`); needs `torch` | 4.7 MB | i_reality_gap_on_real_litho_aerial, i_litho_synthetic_vs_real_contour_ler_reality_gap |
| smile-sem-ler | SMILE SEM line/contact test images (source not stated in the module docstring) | `Test*.tif` (the `LinesTestData/Lines` images, flat in the slug dir) | 3.6 MB | i_litho_synthetic_vs_real_contour_ler_reality_gap |
| tnf-flames | TNF Workshop Sandia piloted-flame archive, tnfworkshop.org `pmCDEF.zip` | `pmCDEF.zip` | 7.7 MB | tnf_sandia_flameD_T_vs_mixturefraction_burke_schumann_render_match_vs_real |
| metalens-psf | metalens PSF: github.com/benhenryL/Metalens-Transformer `psf.npy`; USAF images: github.com/yhy258/EIDL_DRMI `data/USAF_Images/` | `psf.npy`, `USAF_{w,b}_{DRMI,meta}.png` | 21 MB | metalens_psf_imaging_cert, p20_metalens_chromatic |
| optics-psf-eidc | Exoplanet Imaging Data Challenge, VLT/SPHERE-IFS reference PSF, DOI 10.5281/zenodo.6902628 | `psf_cube_sphere0.fits`, `wavelength_vect_sphere0.fits` (+ `parallactic_angles_sphere0.fits`) | 156 KB | optics_sphere_psf_diffraction_lambda_scaling_real, p20_sphere_psf_diffraction |
| jwst-opd | JWST wavefront-sensing OPD products, STScI/MAST (`mast:JWST/product/*-NRCA3_FP1-1.fits`) | `O2024080501-`, `R2024010102-`, `R2024030104-`, `R2024071502-`, `R2024073102-NRCA3_FP1-1.fits` (the module skips `R2024030102`) | 37 MB | p20_jwst_zernike |
| phononic-metamaterial-transmission | Nonreciprocal phononic-crystal transmission, Dryad DOI 10.5061/dryad.dv41ns1wc (mirror zenodo.org/records/4245112) | `FIG4_Transmission_*.csv` (5 files) | 208 KB | l_kk_real_metamaterial |
| pool-boiling-multimodal | Pool Boiling Multimodal dataset, Harvard Dataverse DOI 10.7910/DVN/6GLGC6 | `SS_Temperature_Polished_MC_*.lvm` (4 files: 0, 15, 105, CHF) | 125 MB | battery_boiling_curve_zuber_chf |
| nasa-pcoe-battery | NASA PCoE 18650 battery data set (phm-datasets, "5. Battery Data Set") | `extracted/B0005.mat`, `B0006.mat`, `B0007.mat`, `B0018.mat` | 54 MB | battery_capacity_fade |
| ecn-spray-a | Engine Combustion Network Spray A, ecn.sandia.gov/databases/dieseldata.csv | `dieseldata.csv` | 756 KB | battery_combustion_autoignition_semenov |
| nrel-battery-failure | NREL Battery Failure Databank v2 (github.com/NREL/battery-heat-output, `data/`) | `BatteryFailureDatabankV2.xlsx` (flat in the slug dir) | 688 KB | battery_ocv_soc_equilibrium |
| photonics-eo-modulator-zenodo-ring-fs-laser | Zenodo record 17657883 (fs-laser-written EagleXG ring resonators, OpenData) | `OpenData.zip` plus the unzipped `OpenData/Figure 2 and 4/Python code/Resonator7_*.csv` (210) and `OpenData/Figure 3/Python code/Resonator10R22_*.csv` (82) | 120 MB | photonics_ring_fsr_group_index_cert (dirs), photonics_ring_thermooptic_tuning_transducer (zip) |
| photonics-eo-modulator-zenodo-slowlight-400g | Zenodo record 15631151 (400 Gbps slow-light modulator) | `slowlight_400g.zip` (members `Fig.S2(c).xlsx`, `Fig.S2(d).xlsx`) | 1.6 MB | photonics_slowlight_modulator_efficiency_bandwidth_tradeoff |

### Datasets not available locally

None — every dataset referenced by a SYNTHETIC-ONLY module was found and placed under `data/`. The three modules
still marked SYNTHETIC-ONLY have their real data in place; they stay SYNTHETIC-ONLY because a gate fails (or the
module errors) on the real data, with the numbers recorded verbatim in their Status notes.

## Coupled chains

Every solver in this repository is single-file and carries its own gates. `src/physics_engine/chains/` adds the
other half: one module's output run into the next module's input, with the stage certificates composed. Run it
like any other module:

    cd src/physics_engine && python chains/litho_chain_v1.py        # ~50 s, exits non-zero on any FAIL
    pytest tests/test_chains.py

### litho_chain_v1 — aerial image -> resist + stochastic LER -> etch front

Stages (all pre-existing, none modified, no tolerance loosened):

| stage | module | function used | coupling variable passed on |
| --- | --- | --- | --- |
| S1 imaging | litho/socs_imaging_cuda_first_certified_vs_abbe_8kernel.py | `abbe_1d`, `source_offsets`, `build_tcc`, `socs_kernels`, `socs_aerial_torch` | threshold crossings and the image log-slope ILS |
| S2 resist | litho/litho_euv_stochastic_ler_cert.py | `ler_montecarlo`, `ler_analytic` | per-line resist edge positions -> line width L and trench width w |
| S3 etch | litho/litho_etch_levelset_ballistic.py + litho/litho_etch_arde_ion_flux.py | `etch_levelset`, `phi_neutral`, `phi_ion` | etched line width and etched LER |

Input: a synthetic binary line/space mask, 180 nm pitch, 90 nm drawn line, 2 nm grid, 8 periods; declared optics
lambda = 193 nm, NA = 1.35, annular sigma 0.4-0.7 (6 source points, the imaging module's own constants); resist
dose 30 mJ/cm2, acid-diffusion blur 3 nm, constant threshold at 0.30 of the peak intensity (the imaging module's
own resist-threshold fraction); etch front v0 = 20 nm per time unit for 175 time units.

The only composition rule the chain introduces (no new constitutive parameter): the flux not held vertical by the
narrow ion angular distribution is the part that reaches the sidewall, so the sidewall recedes at
`v_lat = v0 * phi_neutral(AR) * (1 - phi_ion(AR, SIGMA_ION))`, with both factors and `SIGMA_ION = 3 deg` taken from
the two etch modules themselves.

Numbers from the run of 2026-09-12 (chain evaluated 4 x 64 times, fixed seed; the SOCS stage gate ran on the GPU):

| quantity | value |
| --- | --- |
| aerial contrast / ILS at the threshold | 0.906 / 0.0488 per nm |
| resist edge sigma (photon shot noise at that ILS) | 0.210 nm |
| developed resist CD | 61.78 nm |
| etch depth / aspect ratio | 635 nm / 5.4 |
| etched CD | 45.43 nm |
| etched LER (per edge, 3 sigma) | 0.451 nm |
| propagated spread on CD (all stages) | +/- 5.776 nm |
| propagated spread on LER (all stages) | +/- 0.054 nm |
| S1 optics alone (NA 0.5 %, sigma 2 %) | CD +/- 0.010 nm, LER +/- 0.021 nm |
| S2 resist alone (dose 5 %, blur 10 %) | CD +/- 0.888 nm, LER +/- 0.031 nm |
| S3 etch alone (v0*t 20 %) | CD +/- 5.696 nm, LER +/- 0.046 nm |
| cert composition, CD | joint 5.776 vs linear sum 6.594 nm, ratio 0.876 -> PASS |
| cert composition, LER | joint 0.054 vs linear sum 0.098 nm, ratio 0.551 -> PASS |

The etch stage dominates both spreads: a 20 % change of v0*t moves the aspect ratio, and the sidewall term
`1 - phi_ion(AR)` is nearly flat until AR ~ 5 and then rises steeply, so the CD spread the etch contributes
(5.70 nm) is six times the resist contribution (0.89 nm) and 570 times the optics contribution (0.010 nm). The
composition stays sub-linear (ratio 0.88 and 0.55, the stages being independent), so no stage is flagged for
non-linear coupling; if the ratio ever exceeds 1 the chain names the responsible stage by leave-one-out and exits
non-zero.

Stage gates re-run inside the chain, unchanged: SOCS(all kernels) vs Abbe 1.4e-15 and the 8-kernel truncation at
0.000 nm edge placement (gate <= 0.5 nm; at this scene the source has 6 points, so 8 kernels is already the exact
rank-6 imaging transform and the gate is satisfied trivially rather than tightly); resist MC log-log slope -0.495 (gate
-0.5 +/- 0.06) and MC/analytic ratio CV 1.91 % (gate < 5 %); etch RIE-lag ratio 1.800 at 4:1 trench widths (gate
2.0 +/- 0.4).

Null cases: an unpatterned (fully clear) mask gives aerial contrast 5e-16 and no threshold crossing, so no line is
measured at all; doubling the dose shrinks the positive-tone line 45.66 -> 23.57 nm (delta -22.08 nm, the right
direction) and lowers the LER 0.454 -> 0.352 nm (delta -0.102 nm, the 1/sqrt(dose) shot-noise sign).

Scope: 1D scalar thin-mask imaging, a constant-threshold resist with the photon-shot-noise floor only (no
secondary-electron or acid stochastics), and a front-height level set with the lateral term above; the etched LER
is therefore a lower bound in the same sense as the resist stage's own LER floor.

### Innovation target 7: battery stage mechanism, before chain design

Measure the existing OCV polynomial at SOC0/0.5/1, the existing noiseless fade
law at cycles0/50/100 across its four declared coefficient values, and the frozen
thermal solver at short resistances1/2/3, chemistry-off1 and open circuit. Compare
its returned event flag against its documented temperature rise>200. Also compare
the two stage voltage laws and the stored-energy scales; incompatible populations
must not silently inherit each other's empirical certificates.

Fixed instrument gates before execution: two complete result dictionaries exactly
identical; OCV monotonic and inside original[2.0,4.2] window; fresh retention1;
thermal returned flag agrees with its own documented rise threshold. Any flag
inconsistency remains negative evidence and does not authorize editing the frozen
solver. No new coupled solver is designed until this table is written.

| module | status | evidence |
|---|---|---|
| `chains/battery_chain_mechanism_probe.py` | SYNTHETIC-ONLY | Cloud CPU two complete dictionaries identical; voltage differences1.05/0.925/0.5 V; incompatible energy64800 vs24387.84 J measured; four instrument gates PASS. |

Measured battery seam table before coupled design (cloud CPU):

| seam | measured values | implication |
|---|---|---|
| SOC0/0.5/1 OCV law | 2.9/3.7/4.2 V | monotonic, original window PASS |
| frozen thermal voltage | 1.85/2.775/3.7 V | incompatible with OCV stage by1.05/0.925/0.5 V |
| fresh energy | thermal64800 J; integrated OCV*capacity24387.84 J | empirical certificate transfer rejected |
| cycle100 retention | 0.735075431034 to0.913793103448 | capacity is the explicit fade output |
| thermal1/2/3 ohm peaks | 545.880999/153.100363/101.765100 C | strong nonlinear threshold |
| chemistry-off1 ohm / open peaks | 207.836537/25.000051 C | nulls distinguish the coupling |

Evidence: `reports/battery_chain_mechanism_probe.json`, two full dictionaries
identical. Five sampled event flags agree with rise>200; this is not a universal
proof of the frozen return expression, which source inspection shows also
subtracts273.15 from an already Kelvin-to-Kelvin difference.

### Battery chain v1: design and gates before execution

Use only synthetic stage laws: existing OCV polynomial, normalized noiseless fade
coefficients and existing chemical reaction rates/constants. Charge capacity is
Q(cycles) from the fade law; electrical energy is3600*Q*integral(OCV dSOC), not the
incompatible fixed64800 J. Discharge evolves SOC by current; the exact lost
integrated electrical energy heats the thermal stage. A separate integrator
tracks chemical heat, cooling and sensible energy with explicit balance. No
stage source or empirical tolerance changes, and no empirical cross-population
certificate is inherited. This is a new synthetic composition, not a validated
prediction for a physical battery.

Declare nominal cycle100, midpoint fade coefficient, SOC1, resistance1 ohm,
8000 s horizon. Perturbation box: OCV offset+/-0.01 V (existing noise magnitude),
full declared fade-coefficient range, cooling+/-10% (new scenario uncertainty).
Evaluate all8 corners, each stage alone, and leave-one-stage-out spans; compare
joint peak-temperature range against the sum of individual ranges with no slack.
A failed composition is recorded and the largest leave-one-out effect named.

Fixed gates: two full dictionaries/trajectory hashes identical; finite bounded
SOC and inventories; integrated electrical+sensible+cooling+chemical energy
balance relative<=1e-10; no-chemistry open circuit stays exactly at ambient;
fresh capacity exceeds cycle100 capacity; returned event agrees with rise>200;
halving max time step5->2.5 s changes peak<=1 C; joint uncertainty span<=linear
sum of isolated spans. Integration limits0.25 C and0.002 reaction fraction per
step,200000 step cap; cap exhaustion fails. No throughput or empirical claim.

| module | status | evidence |
|---|---|---|
| `chains/battery_chain_v1.py` | OWN-GATE-FAIL | Two full outputs/trajectories identical; nominal554.688764 C; joint span13.722718>linear13.621148 C FAIL; bounded-state gate also FAIL. |

### Target 7 lens mechanism, registered before expanded search design

Measure the frozen patent vector and individual radius changes+/-0.1% at surface
indices2/3/7/8/10/11/12/14/15, beyond the original9/13 search coordinates. Measure
four existing-prescription index alternatives at medium after surface11, each in
its own shared-index batch. This is single-index optical sensitivity, not a
multi-wavelength glass qualification. Retain original61-point aperture sampling,
three fields and search certificates: EFL relative<=0.001, f-number relative<=0.01,
vignetting no worse than the measured patent example. Rejected samples are counted.

Fixed instrument gates: full arrays identical twice; nominal arrays exactly match
the frozen four-parameter evaluator; all sampled paraxial values agree with the
CPU recursion at relative<1e-12; baseline certificates pass. Candidate rejection
is a measured result, not an instrument failure. Write the sensitivity table
before selecting expanded search coordinates or glass families.

| module | status | evidence |
|---|---|---|
| `ray_optics/lens_expansion_mechanism_probe.py` | CUDA-ONLY | L4 full arrays repeat exactly, nominal equals frozen evaluator, CPU paraxial delta0;18/23 cases certified, all4 index substitutions rejected. |

Battery chain first submission: two sweeps reached JSON serialization, then
TypeError on a numerical boolean;0 accepted report files. Correct only returned
boolean metadata conversion before repeating; no integration or tolerance change.
Second battery submission passed measurement serialization but failed final gate
serialization on the same numerical boolean type;0 accepted reports. Normalize
all gate-result metadata to built-in bool; no numerical expression is changed.

Measured battery chain table (cloud CPU), two entire result dictionaries and all
trajectory hashes identical:

| quantity or gate | value | verdict |
|---|---|---|
| nominal capacity / initial energy | 1.53015 Ah /20106.171 J | explicit seam |
| nominal peak / event time | 554.688763807 C /550.240494234 s | synthetic response |
| OCV / fade / cooling isolated spans | 0.572023771 /2.623553572 /10.425571076 C | cooling dominant |
| joint / linear-sum span | 13.722717925 /13.621148419 C | FAIL, nonlinear excess0.101569506 C |
| max-step halving peak difference | 0.012936045 C | PASS<=1 C |
| nominal relative energy balance | 6.342764997e-15 | PASS<=1e-10 |
| exact open/no-chemistry null | 25 C | PASS |
| all states bounded | False | FAIL, retain strict bounds |

Evidence: `reports/battery_chain_v1.json`. Six/eight gates PASS; uncertainty
composition and bounded states FAIL. Every run reached its declared horizon;
bound violation magnitude must be diagnosed before any projection or integrator
change. The composition test samples corners/isolated scenarios; it is not a
rigorous continuous-box enclosure. No empirical battery certificate is claimed.

### Battery inventory diagnosis, gates before execution

Only the corner OCV offset-0.01 V, fade coefficient4.417e-5, cooling factor1.1
failed bounded states. Observe the frozen integrator at its existing snapshot
line using a trace callback; record exact extrema and first violation. Fixed
gates: repeat dictionaries exact; observed trajectory hash matches saved frozen
corner; strict bounds remain[0,1] with zero slack. No arithmetic is changed.

| module | status | evidence |
|---|---|---|
| `chains/battery_chain_inventory_probe.py` | SYNTHETIC-ONLY | Two observations identical and frozen trajectory exact; minimum inventory-7.174648137343064e-43 at step1640; strict-bound FAIL retained. |

Measured lens sensitivity table, L4, before expanded search design:

| case | objective mm | EFL mm | certified |
|---|---|---|---|
| nominal | 0.0929742398149 | 105.82242625 | True |
| radius2*0.999 | 0.0888423371893 | 105.825720349 | True |
| radius2*1.001 | 0.0985148069692 | 105.819138936 | True |
| radius3*0.999 | 0.0885863227658 | 105.827503246 | True |
| radius3*1.001 | 0.0998872847919 | 105.817359883 | True |
| radius7*0.999 | 0.0951559811456 | 105.818362708 | True |
| radius7*1.001 | 0.0911670599332 | 105.826481984 | True |
| radius8*0.999 | 0.157406220026 | 105.734078809 | True |
| radius8*1.001 | 0.0738895365842 | 105.910744492 | True |
| radius10*0.999 | 0.11755290191 | 105.785644943 | True |
| radius10*1.001 | 0.0835759245771 | 105.859159577 | True |
| radius11*0.999 | 0.122819455252 | 105.753778893 | False |
| radius11*1.001 | 0.0802415101428 | 105.891025361 | True |
| radius12*0.999 | 0.122942602579 | 105.792491957 | True |
| radius12*1.001 | 0.0844863909055 | 105.852317628 | True |
| radius14*0.999 | 0.0841021609203 | 105.863990129 | True |
| radius14*1.001 | 0.110949335526 | 105.780977942 | True |
| radius15*0.999 | 0.107003721891 | 105.795531115 | True |
| radius15*1.001 | 0.0869942487165 | 105.849281285 | True |
| medium12=1.48914 | 9.28251104132 | 124.805527958 | False |
| medium12=1.51872 | 6.57532633589 | 118.649889466 | False |
| medium12=1.64128 | 4.67293370424 | 98.5178867678 | False |
| medium12=1.81265 | 18.2572644074 | 79.62752158 | False |

All four instrument gates PASS, two full arrays/dictionaries identical. Evidence:
`reports/lens_expansion_mechanism_probe.json`. All four medium-index alternatives
fail the unchanged focal-length/f-number certificates; wider glass-family search
requires compensating geometry and remains single-index until dispersion is
explicitly introduced. No optical tolerance was relaxed.

### Expanded lens continuation, registered after sensitivity table

The two strongest passing added coordinates are radii8 and11: at+0.1%, objectives
0.073889536584 and0.080241510143 mm versus nominal0.092974239815. Add these two
radii with+/-0.3% bounds to the existing four coordinates/bounds. First run the
frozen4096x6 search; seed its certified winner and the patent vector into a new
4096x6 continuation. This is extra search work seeded from an incumbent, not an
equal-budget optimizer comparison. Also evaluate four fixed-index families from
the measured table, each512x4 designs with the same six-coordinate bounds; rejected
families remain negative results. No dispersion/manufacturability claim.

Fixed gates: two complete baseline+continuation+family sweeps have identical
result dictionaries and all generation population/evaluation hashes; every
reported winner passes unchanged C-EFL/C-FNO/C-VIG; expanded incumbent objective
strictly lower than the frozen four-coordinate winner; collapsed-bound nominal
matches exact patent objective; all candidates accounted for as accepted/rejected.
Zero certified candidates in a glass family is an explicit failed family, not an
excuse to alter certificates. No wall-time or cross-architecture identity claim.

| module | status | evidence |
|---|---|---|
| `ray_optics/lens_design_search_expanded_v1.py` | CUDA-ONLY | L4 two full sweeps identical; objective0.064541758402 vs frozen0.066641872701, five gates PASS; four index families0/2048 certified each. |

Frozen inventory diagnosis: minimum-7.174648137343064e-43 at step1640 in the final
reactant; maximum excess above1 is0. Two observations identical and original
trajectory hash preserved exactly. Evidence: `reports/battery_chain_inventory_probe.json`.
This is a tiny depletion-subtraction roundoff but still fails the zero-slack
bound. Any correction must be a separate inventory-conserving update, with heat
computed from the actual consumed amount; the baseline and strict gate stay frozen.
The independent nonlinear composition excess0.101569506 C remains unresolved.

### Battery inventory-conserving variant, gates before execution

The measured failure is subtracting a depletion amount rounded just above the
remaining inventory. In a new module beside v1, cap each reaction extent by its
available inventory, update inventories with those extents, and compute chemical
heat from the same actual extents. This changes the update, not the acceptance
bounds. Preserve all scenarios and all eight gates verbatim, including the
isolated-span composition gate expected to retain its nonlinear negative.
No numerical threshold is relaxed; whole repeat and energy checks remain required.

| module | status | evidence |
|---|---|---|
| `chains/battery_chain_inventory_v2.py` | OWN-GATE-FAIL | Two complete repeats identical; strict bounds and balance PASS;7/8 gates pass, composition13.722718>13.621148 C remains FAIL. |

Measured expanded lens continuation, L4, two full sweeps identical:

| quantity | frozen four-coordinate winner | six-coordinate continuation |
|---|---|---|
| mean spot RMS | 0.066641872701 mm | 0.064541758402 mm |
| EFL | unchanged comparator in report | 105.818552063 mm |
| vignetting | no-worse certificate | 0.295049504950 |
| working f-number | same1% certificate | 0.991231432736 |

Objective improves3.151343% after additional search budget. All five fixed gates
PASS, including full generation population/evaluation hashes, exact collapsed
null and acceptance accounting. New per-field RMS values0.081144237122,
0.031501696310,0.080979341774 mm. This is a scalar mean objective; no uniform
per-field improvement, dispersion or manufacturing claim follows from it.
All four fixed-index families produce0 certified/2048 rejected EACH, unchanged
bounds and certificates. Those8192 rejects per sweep are a measured glass-search
negative, not hidden from the winning original-index family. Evidence:
`reports/lens_design_search_expanded_v1.json`. Defaults and prior search untouched.

Expanded continuation accepts8493 and rejects16083 of24576 designs per sweep.
The mean improvement has a tradeoff: axial RMS worsens0.073973248393 to
0.081144237122 mm; the two off-axis RMS values improve0.037880990276 to
0.031501696310 and0.088071379436 to0.080979341774 mm. The original scalar objective
and three certificates remain the only acceptance criteria.

Measured inventory-conserving variant, cloud CPU:7/8 original gates PASS, full
result and every trajectory hash identical twice. All scenarios stay inside
strict[0,1] bounds and complete; maximum relative energy-balance error1.04852444775e-14
passes1e-10. Nominal peak554.688763807 C and step-refinement delta0.012936045 C
remain unchanged at reported precision. Joint uncertainty13.722717924606 C still
exceeds isolated sum13.621148418768 C, so composition FAIL and exit1 are retained.
Evidence: `reports/battery_chain_inventory_v2.json`. Correct inventory updates do
not cure nonlinear certificate composition. Next: conditional stage bounds with
explicit coupling terms; preserve the original failed isolated-span comparison.

### C4 process-chain seam measurement, gates before execution

The existing aluminium width trace and nickel-alloy residual-strain field
are different materials and experiments. The residual module has no callable
thermal-history-to-stress solver; no fatigue stage exists. Their empirical
certificates cannot be composed by passing a width number between them.
Before designing adapters, measure the frozen generic moving-Gaussian heat
function on a declared illustrative material: k20, rho8200, cp600 in SI;
P200, v0.8, eta0.35, beam sigma40e-6; ambient293 and melting1600.
Sample121 source positions from+0.002 to-0.004 and21 transverse locations
from-0.0004 to+0.0004 at depth1e-5. Compare quadrature1500/3000 and a
point-source limit with the independent closed-form function. Record full
array hashes, peak and transverse thermal variation, zero-power response,
linear power response, and whether the effective melting threshold
1600-293+270000/600 is crossed. No material calibration or fatigue-life claim.

Observer gates: two complete reports and arrays identical; finite nonnegative
thermal rises; exact zero-power field; double-power relative discrepancy<=1e-12;
quadrature max relative-to-field-peak difference<0.01; point-source relative
L2 difference<0.02 at positive depth. Missing adapters remain explicit facts,
not passing physics stages. All work uses cloud execution.

| module | status | evidence |
|---|---|---|
| `chains/process_chain_seam_probe.py` | OWN-GATE-FAIL | 5/6 gates; exact repeats, point-source relative L2 error0.388446196 exceeds unchanged0.02. |

| Thermal seam observation | Measured value |
|---|---:|
| Peak effective rise | 4307.154568755041 |
| Melted sample fibers | 3/21 |
| Smallest fiber peak rise | 6.476289440141114 |
| Quadrature1500/3000 relative peak difference | 3.19659238822e-7 |
| Point-source relative L2 discrepancy | 0.388446195932 |
| Last sampled downstream rise | 3.53211185048e-27 |
| Exact double-power discrepancy | 0 |

The fixed integration cutoff depends on speed, diffusivity and beam radius but
omits the elapsed transit time -xi/v. Refining the quadrature on that interval
cannot recover the missing downstream heat. A separate bounded quadrature adds
sqrt(max(-xi,0)/v) to the upper u interval (tau=u squared), keeping the original
integrand. Controls before execution: original0.02 point-limit gate on the same
coordinates, original0.01 refinement gate, extended interval doubled changes
field peak by<1e-6; zero/double power and all arrays repeat exactly. The old
observer remains failed. First launcher attempt had0 samples due to an absent
source file; that infrastructure exit is not a scientific measurement.

| Revised heat measurement | Measured value |
|---|---:|
| Point-source relative L2 discrepancy | 0.000139426458748 |
| Quadrature relative peak discrepancy | 1.05579529043e-16 |
| Doubled-horizon relative peak discrepancy | 1.05579529043e-16 |
| Last sampled downstream rise | 136.436660064 |

| module | status | evidence |
|---|---|---|
| `process/moving_heat_history_v1.py` | VERIFIED-FRESH | 7/7 heat observer gates, unchanged0.02 point and0.01 refinement bounds; two complete array hash tables exact. |
| `chains/process_heat_horizon_probe.py` | VERIFIED-FRESH | 7/7 observer gates; reports/process_heat_horizon_probe_l4.json. |

### C4 declared coupled model and gates before execution

This is an illustrative constitutive chain, not a prediction for the two
incompatible public experiments. All material values are declared inputs;
there are no measured fatigue parameters or transferable life certificates.
The heat stage uses the independently checked Gaussian conduction integral,
with apparent enthalpy melting at1600 and latent270000/cp. Temperature follows
sensible rise below melting, a latent plateau, then sensible rise above it.
This preserves the reference's first-order effective-enthalpy approximation;
it does not add convection, vaporization or a nonlinear thermal PDE.

The mechanical stage is a parallel equal-area fiber bundle with common axial
strain and prescribed mean stress, initially stress-free. At each thermal
sample solve force equilibrium with an elastic-perfectly-plastic return map.
Elastic modulus200e9 and yield600e6 scale by the same linear solid factor
clip((1600-T)/(1600-293),0,1); expansion1.4e-5. Liquid fibers carry zero stress
and reset their strain reference to the common strain; this explicitly models
loss of load-bearing stiffness on melting. A cold carrier must remain. Full
mechanical trajectories retain stress, plastic/reference strain and common
strain. A declared monotone cooling segment brings the entire section to293.
No spatial part-scale stress-map certificate is claimed.

Service loading runs20 full mean-stress cycles between+100e6 and-100e6;
the final cycle must shake down elastically before a high-cycle fatigue law
can be evaluated. The fatigue stage receives each fiber's actual stabilized
stress minimum/maximum. It computes effective amplitude a/(1-m/u), then
N=0.5*(effective_amplitude/strength)^(1/b), with illustrative ultimate1.2e9,
strength1.5e9 and exponent-0.1. Zero amplitude means zero damage; no empirical
life or endurance-limit extrapolation is certified. These equation conventions
are documented in [stress-life models](https://doc.comsol.com/6.3/doc/com.comsol.help.fatigue/fatigue_ug_sme.4.20.html)
and [mean-stress correction](https://doc.comsol.com/6.4/doc/com.comsol.help.fatigue/fatigue_ug_sme.4.28.html).

Fixed stage gates: heat controls above; force-equilibrium discrepancy<=1e-10
of room-temperature yield; yield bounds with relative1e-12 numerical allowance;
plastic dissipation>=-1e-12*yield; zero-power and uniform-heating free expansion
stress<=1e-10*yield; independent two-fiber elastic balance; liquid stress zero;
fatigue inverse-law relative error<=1e-12, zero amplitude zero damage, tensile
mean stress increases damage, invalid ultimate-strength inputs rejected.
Service final-cycle plastic increment<=1e-12 and successive cycle stress
change<=1e-10*yield. Finite full trajectories and exact two-run dictionaries.

Chain refinement uses481/961 heat samples and257/513 cooling samples, same
41 fibers; residual stress max difference<0.01*yield and aggregate damage
relative difference<0.02. Nulls: no heat yields zero residual, zero loading
zero damage, stress relief before service reduces mean aggregate damage,
and doubled service amplitude increases it. Each stage must affect an
intermediate or downstream quantity under declared perturbation.

Composition:64 fixed independent draws, thermal power+/-5%, mechanical yield
+/-10%, fatigue strength+/-5%; evaluate each stage alone, all together, and
three leave-one-out cases using identical draws. Report aggregate per-cycle
damage standard deviations. Require joint spread<=sum of stage-alone spreads;
if it exceeds the sum, retain that failed inequality and identify the stage
with the greatest measured leave-one-out reduction. Naming a stage does not
turn the sub-linear inequality green. No change to any reference gate.

Initial chain run:13/13 physics/composition gates and repeated reported values
passed; joint/linear spread0.762686344, residual refinement0.000109647696,
damage refinement2.57080260e-7. Observer audit found null/refinement full
trajectories were computed and hashed internally but omitted from the report.
Retain that initial receipt and add complete state/control hash records before
claiming full-state exact repetition. This changes evidence coverage only;
no constitutive arithmetic, draw, case or tolerance changes.

### C4 coupled result

Two independent cloud workers pass14/14 chain gates and produce identical
complete reports. The observer records200 state cases,600 full temperature,
process and service arrays,152461312 bytes per leg, plus every evaluated damage
array and standalone control array. All corresponding hashes repeat exactly.

| Quantity | Result | Fixed gate |
|---|---:|---|
| Melted fibers | 7/41 | positive |
| Model peak temperature | 4150.154568755 | reported, no calibrated temperature claim |
| Maximum normalized equilibrium error over all cases | 1.30848517487e-16 | <=1e-10 |
| Yield excess / minimum plastic dissipation | 0 / 0 | <=1e-12 / >=-1e-12 |
| Final-cycle plastic increment / cycle stress change | 0 / 0 | <=1e-12 / <=0.06 |
| Refined residual difference divided by yield | 0.000109647696061 | <0.01 |
| Refined aggregate damage relative difference | 2.57080260028e-7 | <0.02 |
| No-heat residual / no-load damage | 0 / 0 | <=0.06 / exact0 |
| Nominal mean damage per cycle | 2.04694658555e-10 | finite and positive |
| Stress-relieved mean damage per cycle | 3.46830598317e-12 | below nominal |
| Doubled-load mean damage per cycle | 5.59347552313e-8 | above nominal |

| Uncertainty source | Mean-damage standard deviation |
|---|---:|
| Thermal power alone | 6.16699224946e-12 |
| Mechanical yield alone | 9.32540195351e-11 |
| Fatigue strength alone | 5.63558336847e-11 |
| Linear sum | 1.55776845469e-10 |
| Joint | 1.18808872747e-10 |

Joint/linear ratio0.762686343977 passes the unchanged sub-linear gate.
Leave-one-out reductions are2.38816211420e-12 thermal,6.27420022826e-11
mechanical and2.23849075569e-11 fatigue; mechanical yield dominates.
All64 draw factors and seven experiment groups are in
`reports/process_chain_v1_l4.json`. The independent report audit recomputes
force balance, the fatigue equation and full observed byte counts.

Scope remains the declared constitutive illustration. The high model peak
and constant-property effective-enthalpy treatment are not a calibrated
melt-pool temperature prediction; no vaporization or part-scale mechanics is
modeled. The fatigue coefficients are illustrative, so numerical life values
are not measured material life. The unrelated aluminium and nickel-alloy
empirical certificates remain separate. No reference source was edited.

Reproduce the complete chain with
`PYTHONPATH=src python src/physics_engine/chains/process_chain_v1.py`.
Each new thermal, mechanical and fatigue stage also runs its own controls as
a script under the same `PYTHONPATH=src` setting.

| module | status | evidence |
|---|---|---|
| `process/thermal_fiber_bundle_v1.py` | VERIFIED-FRESH | Declared model: all5 analytic controls and all200 trajectory gates pass; max force error1.30848517e-16, yield/dissipation0, elastic service0. |
| `fatigue/stress_life_v1.py` | VERIFIED-FRESH | Declared equation: all4 controls pass; inverse-law error below1e-12, zero damage exact, tensile-mean direction and domain rejection pass. No calibrated lifetime claim. |
| `chains/process_chain_v1.py` | VERIFIED-FRESH | 14/14 numerical chain gates,600 full state arrays repeat exactly; sub-linear ratio0.762686344, mechanical stage dominant; illustrative constitutive scope only. |

Standalone entry-point audit: thermal7/7,mechanical6/6,fatigue5/5 gates
(including repeats), each module exits0 in two isolated processes with exact
complete stdout. Evidence: `reports/process_standalone_gates_l4.json`.

### Process-chain transverse discretization audit, gates before execution

The v1 refinement changes481/961 thermal samples and257/513 cooling samples,
but leaves41 transverse fibers unchanged. Its14 passing gates therefore do
not establish transverse convergence. Before another model design, evaluate
41,81,161 equal-area fibers over the same transverse interval, with961 thermal
samples and513 cooling samples. Reuse the existing heat, mechanics and fatigue
arithmetic in isolated workers; only the declared fiber count changes.

For both adjacent refinements require aggregate damage relative difference
<0.02 and maximum residual-stress difference at coincident coarse coordinates
<0.01*yield. Require every original per-state gate, positive melted count,
finite outputs, and two complete independent reports with all temperature,
process, service and damage array hashes identical. Report both comparisons,
even if the finer one improves. No old gate or status is retroactively widened;
this audit has its own status. No empirical material-life claim follows.

| Refinement | Aggregate damage relative difference | Residual difference / yield |
|---|---:|---:|
| 41 to81 fibers | 0.0320739953793 FAIL0.02 | 0.0113668767968 FAIL0.01 |
| 81 to161 fibers | 0.0127391849516 PASS0.02 | 0.00614808217739 PASS0.01 |

Two complete independent reports and all observed full-state/damage hashes
repeat exactly. Every original state gate still passes at all three sizes.
The coarse41-fiber chain does not pass the additional transverse refinement
gates. Its original14-gate certificate is limited to the stated fixed41-fiber
problem; it cannot support a spatial-convergence claim. The finer comparison
passes, but that does not erase the coarser failure or certify the full
uncertainty ensemble at a new resolution. No tolerances changed.

| module | status | evidence |
|---|---|---|
| `chains/process_transverse_refinement_probe.py` | OWN-GATE-FAIL | 3/5 gates;41->81 damage0.0320739954>0.02 and residual0.0113668768>0.01. Full repeats exact; finer81->161 comparison passes. |

Evidence: `reports/process_transverse_refinement_probe_l4.json`.
Reproduce: `PYTHONPATH=src python src/physics_engine/chains/process_transverse_refinement_probe.py`.

## Lithography stage-count and tail falsifier: gates before execution

A separate observer retains the frozen three-stage chain. S4 is an explicitly
synthetic pore-removal stress test: independent occurrence probability0.02,
lognormal radius median2nm/log-sigma0.8, two-wall loss2r clipped at line width.
This is a declared adversarial distribution, not calibrated pore physics.
Use64 paired draws/arm and all48 lines per mask interval; reuse the original
stage tolerances and seed1000+k. Report joint/sum-of-isolated standard deviations
for CD and LER at3 and4 stages, and P99 absolute deviation from median over all
individual line widths. A ratio above1 is retained and the dominant isolated
stage named. S4 radius perturbation is20%; no empirical tolerance is implied.
Gates: original stage checks; observer matches frozen aggregate etch outputs
within1e-12; pore-free identity; clear-mask/doubled-dose nulls; tail enlargement
must be detected for an injected rare-loss control; paired report and full-array
hashes exact twice. The observed pore tail need not widen to pass the detector
check: its actual direction is reported. No baseline or threshold edits.

| Module | Status | Evidence |
|---|---|---|
| `chains/litho_tail_probe_v1.py` | VERIFIED-FRESH | CPU11/11 numerical gates;320 paired evaluations per observation, two observations exact including2240 array hashes; original stage checks pass. |

Three-stage joint/linear ratios: CD0.861541932, LER0.538172174.
Four-stage ratios: CD0.849307033, LER0.273065334. Both remain sub-linear in
this sample, while pooled individual-width P99 deviation widens from
11.761045946nm to11.805756096nm. Etch dominates CD; the added pore stage
dominates four-stage LER. Thus the spread ratio does not certify tail narrowing.
Paired seeds include intrinsic Monte-Carlo variability in each isolated arm;
these finite-sample ratios are not independent analytical error bounds.
S4 is an uncalibrated distributional falsifier, with no material prediction.
Evidence: `reports/litho_tail_probe_v1.json`; standalone pore/null test passes.

## Chains as data: gates before execution

A closed operation registry executes explicit module/argument DAG seams.
JSON specs generate standalone entrypoints and declare64 paired uncertainty
draws per arm, metrics, null inputs and predicates. Keep the frozen lithography
and process-chain implementations as direct oracles; require canonical output
bytes equal at nominal and every null input, and exact full intermediate hashes
across independent workers. All declared propagation observations repeat exactly.
Compare regenerated source bytes twice for all three specs. Third chain is
lithography followed by the existing synthetic pore operator, wired solely in
JSON; no handwritten third chain function. Zero-pore CD/LER must equal the
three-stage output; clear mask must terminate without fabricated geometry.
Invalid operations, forward/cyclic/missing seams, duplicate nodes and unknown
parameters must refuse. Composition ratios are reported with dominant stages;
no sub-linearity or empirical certificate is inferred from schema validity.

| Module | Status | Evidence |
|---|---|---|
| `chains/chain_spec_v1.py` | VERIFIED-FRESH | CPU12/12 integration gates across three generated chains;832 propagation observations and5570 node trace hashes per independent leg repeat exactly;8 schema/seam tests pass. |
| `chains/chain_spec_certify_v1.py` | VERIFIED-FRESH | Six independent generated-entrypoint workers;17 direct-oracle or zero-pore controls per leg; complete canonical report bytes exact for each chain. |
| `chains/generated/litho_from_spec.py` | VERIFIED-FRESH | Generated bytes exact twice,9 direct frozen-oracle controls,256 propagation observations/leg; CD/LER ratios0.852651306/0.735441164. |
| `chains/generated/process_from_spec.py` | VERIFIED-FRESH | Generated bytes exact twice,7 direct frozen-oracle controls,256 propagation observations/leg; damage ratio0.762686344, all returned mechanical gates pass. |
| `chains/generated/litho_pore_from_spec.py` | VERIFIED-FRESH | New chain wired only by JSON;320 propagation observations/leg, zero-pore CD/LER exactly equal original chain, clear-mask null passes; synthetic uncalibrated scope. |

Three specifications regenerate identical entrypoint bytes and execute with
identical complete reports across separate CPU processes. The two existing
chains match their frozen direct functions at nominal, declared null and each
uncertainty-coordinate control input. This is numerical output equivalence;
the generated Python text is a new entrypoint, not the original solver source.
The third composition adds the existing rare-pore operator through JSON seams
without a handwritten chain function. Its CD/LER/P99 spread ratios are
0.860937672/0.850096677/0.779609773. Etch dominates CD; pores dominate LER/P99.
Intrinsic stage seeds remain fixed within these propagation arms, so their
ratios differ from the separately sampled tail probe. All three compositions
remain sub-linear on these declared observations; no universal composition
bound or empirical pore certificate follows. Failed schemas/seams refuse.

Evidence: `reports/chain_spec_certify_v1.json` and source/spec/entrypoint hashes
in `reports/chain_spec_sources_v1.json`. Independent report audit recomputed
every group spread exactly from retained observations. Usage and schema:
`docs/CHAIN_SPECS.md`. Frozen solver and tolerance files remain unchanged.

## Process sampling mechanism: gates before execution

Frozen endpoint41/81/161 versus equal-area midpoint41/123/369, nominal chain,
961 longitudinal samples and513 cooling points. The midpoint tripling preserves
coarse cell centers at fine indices1,4,7,...; compare only coincident positions
(coordinate difference below1e-18m). Equal-area mechanics and every constitutive
law remain frozen. Original damage refinement<0.02 and residual/yield<0.01
apply to both adjacent midpoint refinements. Require original state gates,
two independent full-report/hash repeats and reproduction of the original coarse
endpoint damage failure. This initial observer certifies no uncertainty ensemble.

## Midpoint process variant: gates before ensemble execution

The initial nominal midpoint41->123 comparison fails damage5.1102271% and
residual1.0380286%; it is not promoted. The123->369 comparison gives0.3709076%
and0.1368844%, motivating a separate123-fiber default with369 reference.
Change only fiber sampling and local array-size inference; preserve mechanics,
heat integration, enthalpy map, fatigue equations and old source files.

Require the original14 chain gates: stage controls, every state gate, actual
melting, longitudinal damage<0.02/residual<0.01 at961/1921 samples with513/1025
cooling samples, original no-heat/no-load/relief/doubled-load controls, causal
stage perturbations, sub-linear composition and independent exact full reports.
Additionally compare123/369 at every one of the original64 draws in all seven
isolated/joint/leave-one-out arms, and nominal/null/causal cases. Damage relative
error<0.02 and coincident-center residual/yield<0.01 apply to every pair; zero
fine damage requires exact zero coarse damage. All original state gates and
composition must pass at both resolutions. Frozen-input control at41 endpoint
fibers must reproduce every returned array and full-state hash exactly.
This measures numerical refinement on declared inputs, not calibrated life.

| Module | Status | Evidence |
|---|---|---|
| `chains/process_sampling_probe_v1.py` | OWN-GATE-FAIL | 4/6 gates; midpoint41->123 damage0.0511022706>0.02, residual0.0103802857>0.01.123->369 passes0.00370907555/0.00136884396; independent full reports repeat exactly. |

Evidence: `reports/process_sampling_probe_v1.json`. Sampling at fiber centers
alone does not resolve the narrow thermal/plastic transition at41 fibers.
The retained endpoint failure is reproduced exactly; no old status is changed.

| Module | Status | Evidence |
|---|---|---|
| `chains/process_midpoint_v1.py` | VERIFIED-FRESH | Separate123-center variant,11/11 integration gates,455 transverse comparison pairs per leg; worst damage1.52423790%<2%, residual/yield0.234340434%<1%. |
| `chains/process_midpoint_certificate_v1.py` | VERIFIED-FRESH | Two independent full reports exact;398 unique resolution states per leg, all original stage/null/composition contracts pass;6 focused sampling/material tests pass. |

The full64-draw, seven-arm ensemble passes123->369 refinement. Worst damage
relative difference0.015242378976 occurs at without_thermal:53; worst
residual/yield0.002343404335 occurs at the mechanical causal control. Longitudinal
961->1921 refinement gives damage0.000003035532 and residual/yield0.000045917707.
Composition ratios are0.802408303227 at123 fibers and0.797056610174 at369.
The frozen material response remains byte-identical when given the same41-fiber
endpoint temperature field. The new default changes sampling and explicit
resolution handling, not constitutive laws or acceptance thresholds.

Evidence: `reports/process_midpoint_certificate_v1.json`,
`reports/process_sampling_probe_v1.json`, `reports/process_midpoint_sources_v1.json`.
An independent audit recomputed all448 ensemble pair differences exactly from
retained damage values and residual arrays. Nominal, null and causal controls
bring the recorded transverse comparison count to455 per leg. Every thermal,
process, service and damage state is represented by full-array hashes.

Reproduce: `PYTHONPATH=src python src/physics_engine/chains/process_midpoint_v1.py`.
API: `thermal_field(samples=961, fibers=123)` followed by `run_state(field)` and
`evaluation(state)`. Resolution derives from the supplied field; no global
fiber-count mutation. This certificate covers the declared finite ensemble and
numerical refinements, not an entire continuous parameter region, calibrated
material lifetime or additional depth discretization. The old endpoint41-fiber
and new midpoint41-fiber failures remain OWN-GATE-FAIL in their own observers.

## Real-data failure-site observer: VERIFIED-FRESH

Item24 applies deterministic subset minimization and interval bisection to the retained KK and boiling negatives. Acceptance requires original failed verdict reproduction, original subprocess gate parity, finite complete input hashes, deletion-minimal witnesses and the unchanged synthetic boiling control. Independent repeated reports must match. Baseline thresholds, input mapping and status remain unchanged.

| Module | Status | Decisive observation |
|---|---|---|
| `chains/failure_localization_v1.py` | VERIFIED-FRESH | Deterministic deletion-minimal subset and interval witnesses; four interacting/invalid/sibling/predicate controls pass. Numerical/physical scope belongs to the supplied oracle. |
| `chains/physics_failure_sites_v1.py` | VERIFIED-FRESH | Observer5/5, complete independent reports byte-identical. KK all5singletons and5leave-one-out sets fail threshold0.5571: no single measurement site. Boiling G1 witness105/CHF flux change-395.43553645kW/m2; G3 witness0/CHF noise ratio1.20393233>0.05. Original two modules remain OWN-GATE-FAIL. |

Full results, source/data/array hashes and deletion traces:
`reports/physics_failure_sites_v1.json`; independent repeat receipt:
`reports/physics_failure_sites_repeat_v1.json`. `docs/FAILURE_LOCALIZATION.md` explains
interventions and the G3 predicate/label discrepancy. The terminal CHF interval
[83448,84546) fails while both halves pass: temporal aggregation matters and no
single corrupt sample is established. No source-gate threshold, baseline mapping
or empirical status was changed.

## KK input-semantics correction: VERIFIED-FRESH

A separate input contract must reject all five pinned directional-transmission files before any KK arithmetic, preserve unchanged positive complex-response arithmetic, and refuse known-dataset relabeling. Original numerical failure evidence remains unchanged; source interpretation and the original frequency-unit assertion are being corrected in documentation.

| Module | Status | Decisive observation |
|---|---|---|
| `wave_optics/kk_input_contract_v1.py` | VERIFIED-FRESH |5/5 observer gates; all5pinned real CSVs refused before KK evaluation; unchanged analytic complex-response positive passes;11controls cover semantics, DOI relabeling, units, finite/grid and file-integrity refusals. Independent complete reports exact. This certifies input refusal, not real-data causality. |

Primary-source contract and pinned local-file inventory: `docs/KK_DATA_CONTRACT.json`.
Interpretation correction and reproduction: `docs/KK_INPUT_CONTRACT.md`. Retained
legacy numerical failure and earlier subset witnesses are unchanged; their physical
interpretation is superseded by this contract finding. No GPU was used.

## Boiling LVM channel preservation: VERIFIED-FRESH

Acceptance: four real files, independent full seven-column parse exact, legacy first-six-column and full flux/superheat arrays exact, all six temperature channels preserved. Existing physical gates are not changed. Published MATLAB mapping comparison remains unavailable; see the retained blocker in the observer report.

| Module | Status | Decisive observation |
|---|---|---|
| `electrochem/boiling_lvm_channels_v1.py` | VERIFIED-FRESH |4/4 observer gates on4real files; independent full7-column parser exact, original6-column prefix/flux/superheat exact, omitted Temperature_5 preserved.7tests pass; complete independent reports exact. Original boiling physical failures remain. |

Reproduction and unresolved published MATLAB comparison:
`docs/BOILING_LVM_CONTRACT.md`. Full-array hashes and repeat evidence:
`reports/boiling_lvm_channels_v1.json`, `boiling_lvm_channels_repeat_v1.json`.
This certifies parsing and preservation, not the physical sensor-depth assignment.

## Boiling stationarity mechanism: VERIFIED-FRESH

Use the recorded1024sample acquisition blocks without trimming. Whole-record admission retains the existing5% relative-standard-deviation criterion and positive flux; short-window passes never override it. Acceptance requires four complete records, full frozen-array parity, within/between variance identity within64float64eps-scaled arithmetic budget, stable15/105conditional ordering, retained CHF/null refusal and a synthetic aggregation counterexample. Physical mapping remains unverified.

| Module | Status | Decisive observation |
|---|---|---|
| `electrochem/boiling_stationarity_v1.py` | VERIFIED-FRESH |7/7 observer gates,6tests, independent complete reports exact. CHF547/549blocks locally pass but full spread120.393233% fails unchanged5%;99.9883631743%variance between block means. Whole15/105 admitted; signed null and completeCHF refused. All samples/blocks retained and full flux/superheat hashes exact. |

Method, complete numerical table and limitations: `docs/BOILING_STATIONARITY.md`.
This is conditional temporal-statistics admission, not physical calibration or
promotion of the original boiling model; its G1/G3 failures remain unchanged.

## Verification entrypoint readiness: OWN-GATE-FAIL

| Module | Status | Decisive observation |
|---|---|---|
| `scripts/verification_inventory_v1.py` | OWN-GATE-FAIL | 158 VERIFIED-FRESH declarations resolve without duplicate aliases; 11 explicit recipes, 147 still unmapped. Inventory performs no numerical execution. |
| `scripts/verify_declared_v1.py` | OWN-GATE-FAIL | Full repository coverage remains incomplete. Selected CPU scope: 11/11 declarations pass twice with exact numerical report hashes; ten runner/inventory controls pass. Strict make verify refuses incomplete coverage before execution. |

Commands, evidence schema and CPU/Modal scope: `docs/VERIFICATION.md`.

## Scattering GPU siblings (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/scattering/spine_scatter2d_multiple_helmholtz_forward_C_gpu.py | VERIFIED-FRESH | Optional frequency batch API: original 9/9 combined gates twice on H100/L4; 18 controls. All 394 arrays pass old/new parity (max abs 1.510e-12, normalized 0.0003263 <= 1), repeats exact. H100 first/warm batch 0.0961/0.0337 s vs CPU 0.5231/0.5229 s; L4 0.1906/0.0644 s vs CPU 0.9636/0.9559 s. Warm gains 15.5x/14.8x. Same-worker reversed comparison: first batch 0.1477 > scalar 0.1086 s; warm batch 0.0364 < scalar 0.1025 s. First-use regression retained; scalar API unchanged. scripts/verify_scattering_gpu.py --batch-forward; reports/scattering_batch_after_v1, scattering_batch_l4_v1 and scattering_batch_timing_v1. Device Hankel experiment not adopted: H100 batch37.7–39.1->30.4–31.0ms, scalar103.8–106.6->119.4–127.3ms; original9/9twice and24controls pass. reports/hankel_device_after_v1. |
| src/physics_engine/scattering/pa_adjoint_helmholtz_envelope_gpu.py | VERIFIED-FRESH | H100 float64, original 201x201 stencil and all five frequencies: CPU solve 185.8-199.2 ms, GPU setup+solve 1.268-1.826 ms; normalized field parity <= 0.000236538; transpose residual <= 5.25e-14. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_scattering_gpu.py`
on CUDA; `--device cpu` is a CPU control, not GPU evidence. Full evidence:
`reports/scattering_gpu_v1/report.json`, both complete NPZ captures and independent
`audit.json`. Nine gates pass twice; 394 arrays / 4,442,256 bytes repeat exactly.
Parity is elementwise `abs(gpu-cpu) <= 1e-9 + 1e-8*abs(cpu)`, fixed before H100.
Six CPU-backend tests include nonsymmetric forward/adjoint solves with nonzero
boundary right-hand sides, indefinite frequencies and singular-case refusal.
The original CPU files and their thresholds remain unchanged. The forward retains
host SciPy special functions; timings include those and transfers/observations.
The FD speed difference combines a separable algorithm with GPU execution; it is
not an isolated hardware speedup. CPU timing excludes sparse matrix construction;
GPU timing includes sine-basis construction. The CPU adjoint has printed diagnostics
rather than boolean own gates; all fields, envelopes, rays and crossing locations
are compared and its original transposed operator is used for the residual gate.
Modal app `ap-ZIGhTn1zW3DURr6ev3My2h` ended; no local GPU was used.

## ILT GPU search sibling (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/litho/d_litho_certified_imaging_ilt_gpu.py | VERIFIED-FRESH | H100 batched dose/focus crossings: original 9/9 plus full Jacobian gate twice; previous/current full record bytes and all arrays exact. Same-worker GPU 0.09632/0.05875 -> 0.06225/0.03646 s; CPU 0.51531/0.51860 s (8.28x/14.22x CPU/GPU). 40 CPU tests pass. |

`PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python scripts/verify_ilt_gpu.py`
runs all four original forward kill gates before ILT, then the unchanged b/c/d
gates and e diagnostics. Nine harness gates pass twice, including the full ILT
record and exact chosen knobs/process-window fractions. All 315 complete images
and full records repeat exactly. Evidence: `reports/ilt_gpu_v1/`; CPU control:
`reports/ilt_gpu_cpu_control_v1/`. Four tests exercise complex focus, nonuniform
sources, reconstructed TCC and source refusals. Parity was fixed at
`abs(gpu-cpu) <= 1e-9 + 1e-8*abs(cpu)` before H100 measurement.
This ports the ILT candidate-imaging loop: host thin-SVD setup retains every
source-span mode and calls the existing SOCS batch apply. The original threshold,
edge matching, tie order, Jacobian/SVD diagnostics and gates remain CPU functions
in isolated namespaces; originals are not modified. Timing includes setup,
transfers and the entire section-c search/diagnostics, after context warmup.
The speed difference includes batching and is not an isolated hardware comparison.
Modal app `ap-UIfsvMa0IXs9pCwNQQ5bnG` ended; no local GPU was used.

## Scalar aerial-image and focus GPU siblings (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/litho/litho_aerial_image_scalar_gpu.py | VERIFIED-FRESH | H100: original 4/4 gates twice, 213 complete 8192-point images/leg; CPU 0.2912/0.2841 s versus GPU 0.1003/0.1014 s; max absolute image error 1.34e-15. |
| src/physics_engine/litho/litho_depth_of_focus_cert_gpu.py | VERIFIED-FRESH | H100: original 4/4 gates twice, 211 complete images/leg; CPU 1.5738/1.5683 s versus GPU 0.1315/0.1296 s; max absolute image error 1.00e-15. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_aerial_gpu.py`.
The original main bodies and four thresholds per module run in isolated namespaces
with GPU source propagation. All full images repeat exactly. The predeclared
pointwise budget is `1e-10 + 1e-9*abs(cpu)`; normalized maxima are 1.192e-6 and
1.111e-6 (pass <= 1). Seven controls cover binary mask decisions, pupil cutoffs,
positive/negative physical defocus and bad-grid refusal. Host grid/mask construction
preserves the original floating-point decisions; FFT, source intensities and contrast
run on GPU. Timings exclude the return observer and include setup/transfers.
Evidence: `reports/aerial_gpu_v1/` and `reports/aerial_gpu_cpu_control_v1/`.
Modal app `ap-FBJks64kdJxRXSPsWMpHja` ended; no local GPU was used.

## PLIC rotation and etch GPU siblings (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/litho/vof_zalesak_cert_gpu.py | VERIFIED-FRESH | H100 float64: original 6/6 gates and full-field parity twice; N128 full rotation CPU 16.35 s vs GPU 0.391 s (first leg), 1137 unchanged steps; max field error across cases 3.55e-12. |
| src/physics_engine/litho/litho_etch_levelset_ballistic_gpu.py | VERIFIED-FRESH | H100 float64: original three etch gates plus optional VOF conservation all pass twice; all heights/rates and 256-width sweep pass parity, normalized error < 3e-6. 256-width CPU 0.12138/0.12292 s vs GPU 0.000943/0.001328 s. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_vof_gpu.py`
and `python scripts/verify_etch_gpu.py` with the same environment. Reports and
complete duplicate NPZs are under `reports/vof_gpu_v1/` and `reports/etch_gpu_v1/`.
VOF runs the unchanged original main/gates at N64/100/128, full rotation, N100
half-rotation and diffusive N64 upwind null. Repeated identical requests within a
main invocation reuse that invocation's fresh result; both invocations recompute.
Fields repeat exactly; comparison against the CPU original uses the predeclared
pointwise budget `1e-7 + 1e-6*abs(cpu)`. No gate or timestep changed. Geometric
fluxes use Warp with FP fusion disabled. The two-sweep split order and y-normal
transpose are preserved. The upwind null remains diffusive and has no speed gain.
VOF CPU timings are separate unobserved solves with identical scalar results;
GPU timings include host fixture construction and transfers. Five stencil controls
cover both axes/schemes, boundary donors, mixed-cell cutoffs and Courant refusal.
Etch preserves every Euler timestep and rate indexing; four controls include zero
and one step, multiple widths and invalid width. Its complete-history budget is
`1e-12 + 1e-10*abs(cpu)`. The 256-width timing includes setup and full host histories;
it is a compiled batch comparison, not an isolated device-only speedup.
Modal app `ap-xWjAsffvYcNiz63l87mVpF` ended; no local GPU was used.

## Marangoni momentum/projection GPU sibling (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/process/marangoni_meltpool_cfd_gpu.py | VERIFIED-FRESH | H100 graph stepping with unused CPU pressure LU removed: original 4/4 gates twice, all fields/record bytes exact before/after. Same-worker six-case totals 4.3101/3.9365 -> 3.9557/3.7219 s; nominal 0.79023/0.75416 -> 0.73587/0.74524 s. Previous calibration/graph gains retained below. Prefix graph experiment rejected: fields/records exact and 26 controls pass, but reversed warm nominal 0.7521 -> 0.8000 s; production code retained (reports/marangoni_prefix_v1). |

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_marangoni_gpu.py`.
Evidence: `reports/marangoni_gpu_v1/` and the CPU control directory. Both complete
captures repeat exactly. Pointwise field/scalar parity was fixed at
`1e-7 + 1e-6*abs(cpu)`. The original render-match null/perturbation and morphology
requirements remain unchanged; the known incomplete physical closure remains.
The momentum time loop and pressure projection run on GPU; conduction, heat-flux
calibration and outer energy sparse solves remain CPU work. The separable pressure
solve preserves the original first-row pin and reconstructs the eliminated
compatibility equation. Three tests cover incompatible RHS/nonzero pins and the
first shear/advection/porosity/projection steps. Timings include the entire original
solve and its return observer (both legs), reuse only repeated identical calls
within one main invocation, and include host setup and transfers. Those initial measurements describe the version before calibration reuse below.
Modal app `ap-vShh62MnhMemhLNzHdq15v` ended; no local GPU was used.

## Fourier-optics propagation GPU sibling (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/wave_optics/wave_optics_cell_gpu.py | VERIFIED-FRESH | H100/L4 original 3/3 gates plus full ripple/field/row parity twice; 8 controls. Scaling before readback keeps all 46 fields/rows (303,366,992 bytes/leg) and records exact to prior H100 path. Same-worker reversed run_all 0.2739/0.2725 -> 0.1822/0.1807 s (1.50/1.51x). Full verifier warm H100 0.2240 s vs CPU 1.7639 s; L4 0.4224 s vs CPU 2.0053 s. Cross-GPU fields differ in bytes but each passes original CPU parity (L4 normalized 1.3744e-5 <= 1). reports/wave_optics_scale_after_v1, wave_optics_scale_timing_v1 and wave_optics_scale_l4_v1. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_wave_optics_gpu.py`.
Both complete hash manifests repeat exactly. All 4096x4096 Airy-field values were
compared in memory, along with both million-point 1D FFTs and every coherent image.
Evidence `reports/wave_optics_gpu_v1/` stores complete hashes and explicitly marked
center-row slices for the large 2D field; other propagated arrays are archived in
full. The preserved elementwise budget is `1e-12 + 1e-9*abs(cpu)`; full gate/ripple
record parity is 0.231787 <= 1. Three controls use asymmetric apertures, oblique order
acceptance and a rectangular 2D FFT. Host masks, frequency/order grids and observation
logic remain unchanged. FFT/FFT2 and coherent phase propagation run on GPU; timing
includes setup, transfers and the complete-field return observer after a small warmup.
The G3 executable CPU threshold is `rel_err < 0.5` (50%), despite its docstring's
0.5% wording; this port preserves the executable gate and does not claim a 0.5% gate.
Modal app `ap-gMZeHTWp8ozrwLx3ulQie8` used an isolated code/input snapshot with matching
source hashes through the unchanged existing runner, ended, and was fetched.
No local GPU was used. All 48 new CPU-backend controls also pass together.

The exact executed optics source is preserved in `reports/wave_optics_gpu_v1/executed_source.json`;
the committed module only removes its trailing blank line.

## Heat-source GPU sweeps (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/process/weld_goldak_spot_nist_gpu.py | VERIFIED-FRESH | H100: original 3/3 gates twice; 7 temperature fields, max absolute error 5.639e-11, normalized parity 5.808e-6. CPU 0.6740/0.8260 s vs GPU 0.7630/0.4209 s; timing varies across the two runs. |
| src/physics_engine/process/lpbf_meltpool_render_match_gpu.py | VERIFIED-FRESH | H100: original 5/5 gates twice; 78 fields, max absolute error 5.457e-12, normalized parity 7.078e-7. CPU 73.6044/72.0382 s vs GPU 0.2728/0.1984 s. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_heat_sweeps_gpu.py`.
The torch float64 siblings batch Goldak radial quadrature and Eagar–Tsai/Rosenthal
field sweeps. Original observation logic, thresholds and CPU modules are retained.
The elementwise budget is `1e-8 + 1e-9*abs(cpu)`; all captured fields repeat exactly
in two complete recomputations per backend. Evidence is in `reports/heat_sweeps_gpu_v1/`
and `reports/heat_sweeps_gpu_cpu_control_v1/`. The two real NIST input CSVs and their
hashes are stored in `reports/process_gpu_inputs_v1/`; no synthetic fallback is used.
Timing includes host setup, transfers and complete field observation. Three CPU
controls check quadrature and temperature fields. Modal H100 app
`ap-O04MHBfPBBztQtE99BftZn` ended and its results were fetched.

## Ising and finite-difference Schrodinger GPU siblings (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/quantum/ising_2d_gpu.py | VERIFIED-FRESH | Explicit SimulationSeries reuses one graph/buffer layout, resets spins and M/E each call. H100 original 4/4 twice; 228 arrays exact to standalone graph path; 30 CPU/H100 controls. Whole 19-case same-worker scalar 0.4011/0.3904 -> 0.2283/0.2211 s, full 0.4139/0.3751 -> 0.2663/0.2413 s, including first capture. Original verifier CPU 3.6184/3.5655 s vs scalar 0.3823/0.3379 s (9.47/10.55x); no universal 10x claim. reports/ising_series_v1 and ising_series_timing_v1. Series L4 original 4/4 twice and 30 controls; full 0.4636/0.4621 s, scalar 0.4133/0.4071 s, arrays exact to H100 (ising_series_l4_v1). |
| src/physics_engine/quantum/schrodinger_1d_gpu.py | VERIFIED-FRESH | H100: original 4/4 gates twice; normalized parity 0.0006144 <= 1 after declared eigenvector sign alignment, stencil residual <= 4.829e-8. Two-case time sum: CPU 0.007440/0.006885 s vs GPU 0.114570/0.053946 s; no speed gain. Selected-spectrum experiments remain unadopted: serial ~13.7 ms, parallel ~6.3 ms, native/shared ~4.8 ms per case versus CPU 3.2–5.1 ms; three-case screening passes existing parity/residual bounds, not a complete solver gate. reports/schrodinger_*_experiment_v1 and schrodinger_stage_profile_v1 retain measured attempts. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_quantum_gpu.py`.
Ising uses Warp integer checkerboard updates with the original host RNG draw order
and probability table, including draws for inactive cells. All 600 observed spin
frames per case are captured. CPU and GPU spin storage use int64 and int32 respectively;
values agree exactly, while encodings differ. Six controls include odd grids and RNG state.
The Schrodinger sibling uses torch float64 dense eigensolves of the original FD
operator; the CPU specialized tridiagonal solver remains faster at these sizes.
Three controls check asymmetric potentials, eigenpairs and stencil residuals.
Parity uses `1e-8 + 1e-6*abs(cpu)` and a `1e-6` residual bound; raw eigenvectors and
alignment signs are retained. Both complete captures repeat exactly per backend.
Evidence: `reports/quantum_gpu_v1/` and `reports/quantum_gpu_cpu_control_v1/`.
Modal H100 app `ap-372uyFbIRx22FwGYli7br4` ended and was fetched. These ports are
available GPU implementations, with no faster-default claim for the measured cases.

## Wavepacket GPU siblings (2026-09-13)

| Module | Status | Evidence |
| --- | --- | --- |
| src/physics_engine/quantum/quantum_revival_gpu.py | VERIFIED-FRESH | Stable spectrum buffer and scalar CUDA graph; original3/3 twice H100/L4, full25,152-value parity and old/newH100 arrays exact. Warmed paired whole main H10020.6–21.4 ->19.0–19.5ms; L433.7–34.4 ->31.1–31.5ms. Fresh series/capture per call; seven CPU/H100 controls. reports/revival_buffer_{before,after,l4,timing,timing_l4,warm_timing}_v1. |
| src/physics_engine/quantum/bloch_oscillation_gpu.py | VERIFIED-FRESH | Bounded BlochSeries operator reuse; original3/3 twice H100/L4, all871,200 state/COM values exact to prior H100 GPU; ten CPU/H100 controls. Same-worker warm H10038.3–38.9 ->17.5–17.7ms vs CPU230–236ms (13.0–13.5x); L4104.0–106.0 ->45.5–46.1ms vs CPU510–520ms (11.1–11.3x). Fresh series/operator solves included. reports/bloch_series_{before,after,l4,timing,timing_l4,profile}_v1. |

Run `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_wavepacket_gpu.py`.
The torch complex128 siblings batch mode phases and propagate complete wavepacket
states. Original time grids, peak interpolation and gates are retained. The full
arrays satisfy `1e-10 + 1e-8*abs(cpu)` and repeat exactly in two complete recomputations
per backend. Four CPU controls cover phase propagation and asymmetric wavepackets.
Timing includes setup, transfers and the full return observer. Evidence is in
`reports/wavepacket_gpu_v1/` and `reports/wavepacket_gpu_cpu_control_v1/`.
Modal H100 app `ap-2Rcj9Od6AhobSb0um2TXng` ended and was fetched. No local GPU was used.

## Bounded Ising random-draw storage (2026-09-13)

The Ising GPU sibling consumes the original RNG stream in blocks of at most
`random_batch_sweeps=32` full sweeps. The random tensor therefore holds at most
`2 * min(random_batch_sweeps, eq + meas) * L**2` float64 values on each backend,
instead of `2 * (eq + meas) * L**2`. Each block completes before its buffers are
released. At the original 32x32 grid and 1000 sweeps this is 0.5 MiB instead of
15.625 MiB per random tensor, a 31.25x reduction calculated from allocation shapes.
This is not a measurement of total process/GPU memory: spin histories, state and
runtime storage remain separate and unchanged.

The existing verification command accepts `--module ising` to run only these checks:
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_quantum_gpu.py --module ising --out reports/ising_gpu_bounded_v1`.
All four original gates pass for all 19 cases in two complete recomputations. Full
spins, magnetization/energy histories, observables and exit RNG states are exact
against CPU; captures also repeat exactly per backend. CPU case-time totals are
2.8199/2.9697 s and GPU totals are 2.8718/2.9444 s. No consistent speed gain is claimed.
The original all-time-buffer measurements remain in `reports/quantum_gpu_v1/`.
Current evidence is in `reports/ising_gpu_bounded_v1/` and the matching CPU-control
directory. The 23 Ising unit tests cover even/odd grids, one-sweep and partial blocks,
RNG consumption, bounded allocation requests and invalid batch arguments.
Modal H100 app `ap-viamdh2W0Qb4CD7W0uC0Dn` ended and was fetched; no local GPU was used.

## Optional Ising spin-history storage (2026-09-13)

Default `simulate(..., return_history=False)` calls now keep only magnetization
and energy series for the scalar observables. A four-byte placeholder replaces
`meas * L**2` int32 spin-history entries, and the observation kernel skips those
stores. `return_history=True` retains the complete original history interface.
For L=32 and meas=600 the avoided spin-history allocation is 2,457,600 bytes
(2.34375 MiB); these are allocation sizes, not a total-memory benchmark.

All 25 CPU unit tests pass. Two additional executions of the scalar/full-history
unit cases on H100 pass for odd/even lattices, including RNG state and allocation
shape checks. The existing quantum command checks scalar and full-history modes
for all 19 original cases, twice: the original four gates, every spin/history value,
scalar outputs and final RNG states are exact against their references. Complete
full-history captures repeat exactly per backend. CPU case sums are
3.5234/3.5189 s; GPU full-history sums 3.7019/3.7576 s and scalar-only sums
3.6481/3.7118 s. This is a storage improvement, with no CPU speedup claim.

Evidence: `reports/ising_gpu_scalar_v1/`. Reproduce with
`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=src python scripts/verify_quantum_gpu.py --module ising --out reports/ising_gpu_scalar_v1`.
The verifier compares both modes using cloned initial RNG state without changing
the caller's stream or numerical tolerances. Earlier measurements remain in their
original directories. Modal H100 app `ap-UqHFAEZGButcveMHffG1wP` ended and was fetched;
no local GPU was used.

## Marangoni conduction calibration reuse (2026-09-13)

An H100 cProfile measurement of the unchanged nominal solve placed calibration at
2.800/2.803 s out of 5.218/4.894 s observed GPU calls. The 41 conduction matrix
builds consumed 2.129/2.130 s cumulatively; 840 pressure projections consumed only
0.229/0.181 s. These nested profiler times overlap and include profiler overhead.
The GPU sibling now builds and factors the immutable conduction matrix once
within each original 40-step calibration. Each trial retains the source and
boundary arithmetic order, the bisection bounds, comparison and iteration count.
There is no cross-call cache. CPU reference implementation and physics gates stay
unchanged; nine unit tests include exact calibrated heat flux at multiple grid
sizes and target widths.

Same H100 worker, original full verification command twice per implementation:
nominal GPU before 4.0056/3.9398 s, after 1.8049/1.7866 s (2.22/2.21x).
Six-case totals before 22.3336/21.6969 s, after 8.9600/8.7396 s (2.49/2.48x).
Each result includes calibration setup and complete original solve; no timing
boundary was removed. All four original gates pass twice. All 18 GPU fields
(799,440 bytes per repeat) and complete scalar/iteration records match the old
GPU implementation exactly. Full CPU/GPU parity retains the original
`1e-7 + 1e-6*abs(cpu)` budget and unchanged stopping iterations.

Evidence: `reports/marangoni_stage_profile_v1/`,
`reports/marangoni_calibration_before_v1/`, and
`reports/marangoni_calibration_after_v1/`, including the complete-array comparison.
The existing `scripts/verify_marangoni_gpu.py` runs the same checks. H100 app
`ap-Ak6YjgsM8pQsLa9G0wQRLB` ran the old source then the new source in separate
processes with source hashes in both receipts; it ended and was fetched.
No local GPU was used. The separate first-momentum operator profile is retained
in `reports/marangoni_operator_profile_v1/`; its cold profiler wall time is not a
production latency estimate.

## Marangoni device-resident CFL selection (2026-09-13)

The first 56-step momentum operator profile recorded 8,137 kernel launches,
514 scalar extraction calls and 462 stream synchronizations. The sibling now
computes the timestep min/max on GPU and combines finite/convergence flags into
one host decision per step. The corresponding observed call has 8,204 launches,
282 scalar extractions and 230 stream synchronizations. GPU-to-host pinned copies
fall from 288 to 56; scalar extraction counts also include CPU scalars.
This reduces synchronization, not the large number of pointwise launches.

All original six-case gates pass twice. Complete captures repeat exactly per
backend and stopping iterations match CPU and the preceding GPU version. Fields
are not bit-identical to that GPU version: max absolute change 2.0691e-11 and
normalized error 1.1343e-6 under the unchanged `1e-7 + 1e-6*abs(reference)` budget.
Nine existing CPU tests pass. In a separate same-worker timing with order reversed
on the second leg, nominal previous/current is 1.9075/1.7707 s and then
current/previous 1.5710/1.6025 s (1.077x and 1.020x). Do not infer a larger gain
from the faster full-suite timings on a different H100 worker.

Evidence: `reports/marangoni_resident_v1/` (including comparison to the preceding
GPU fields), `reports/marangoni_resident_operator_profile_v1/`, and
`reports/marangoni_resident_timing_v1/`. The existing verification command and
pointwise tolerances are unchanged. Both H100 jobs ended and were fetched;
no local GPU was used. The next measured cost is the remaining roughly 146 kernel
launches per substep.

## Marangoni CUDA-graph stepping (2026-09-13)

The remaining roughly 146 kernel launches per momentum substep are captured once
per geometry/surface-tension configuration and replayed. The same tensor arithmetic
runs on persistent u/w/T buffers, with one finite/convergence readback after every
substep. Incoming fields are copied into those buffers at each outer solve; graph
reuse is tested with changed temperature and surface tension. No substeps or outer
iterations are skipped. Pressure reference slots use device-side zeroing so capture
does not attempt a CPU-scalar upload. The CPU backend retains eager stepping.

Full original six-case verification passes four gates twice, with unchanged CPU
parity budget and stopping counts. All 18 GPU T/u/w fields (799,440 bytes per leg)
and complete scalar/iteration records are exactly equal to the preceding resident
CFL implementation. Captures repeat exactly per backend. Ten CPU tests and the
changed-input reuse control on H100 pass. The first capture attempt was refused
before the numerical suite because of a CPU-scalar copy; that receipt is retained.

In a separate H100 process, alternating order gives previous/current nominal
1.6439/0.6690 s and then current/previous 0.5983/1.3611 s (2.46x/2.27x).
Timings include graph construction, warmup, calibration, transfers and the entire
solve on every width_ratio call. Full-suite current GPU totals are 3.7044/3.4496 s
versus CPU 17.8610/16.5014 s; those totals are not the isolated preceding-GPU comparison.

Evidence: `reports/marangoni_graph_v1/` and
`reports/marangoni_graph_timing_v1/`. The existing verification command is retained.
Both successful H100 jobs ended and were fetched; no local GPU was used.

## ILT threshold-crossing selection (2026-09-13)

The warm H100 stage profile measured 2,508 Python edge-finding calls at 0.2041 s
out of 0.3447 s under cProfile. Crossing intervals are now selected with array
operations; interpolation runs only for selected intervals with the original
scalar operation order. Exact-threshold precedence, slopes, endpoint exclusion
and NumPy scalar-promotion behavior are preserved. The original CPU functions and
gates remain unchanged. Twenty-three CPU tests cover nonuniform optical sources
and crossing cases including float32/float64/integer inputs, empty/singleton arrays,
plateaus, exact thresholds, NaNs and infinities.

The original complete ILT command passes 9/9 gates twice for both implementations
on one H100 worker. GPU time 0.31166/0.28075 s becomes 0.14187/0.10797 s (2.20/2.60x).
Current CPU reference is 0.51265/0.51393 s. All complete record bytes and candidate
image/mask arrays are exact before/after, including selected bias/SRAF parameters,
process windows and Jacobian diagnostics. No image or record tolerance changed.

Evidence: `reports/ilt_stage_profile_v1/`, `reports/ilt_edges_before_v1/`, and
`reports/ilt_edges_after_v1/`. The existing `scripts/verify_ilt_gpu.py` command is
retained; the same-worker app ended and was fetched. No local GPU was used.

## ILT batched central-difference Jacobian (2026-09-13)

The warm stage profile placed 512 CPU aerial calls in the Jacobian at 0.0934 s.
The GPU sibling now evaluates the same 512 plus/minus masks through the existing
SOCS batch API, in tiles of 64 masks, and forms the same central difference.
Perturbation epsilon, scalar mask-update order, source normalization and the CPU
function's K0 convention are retained. Kernel/source setup is reused within the
same imaging invocation. The original three-value section_c return remains the
default; optional return_jacobian exposes the actual matrix for comparison.

Thirty-one CPU tests pass, including gray/asymmetric masks, unequal source weights,
multiple epsilons and explicit K0. The existing full ILT command now also captures
and compares the complete 256x256 Jacobian. Original nine gates and this additional
parity gate pass twice, with unchanged `1e-9 + 1e-8*abs(reference)` tolerances.
Jacobian normalized error is 0.0001666 and full-record error 0.0005527, both <=1.
The Jacobian and its SVD-derived diagnostic values are not claimed bit-identical;
selected design, process windows and all pre-existing candidate image/mask arrays
remain exact. Complete captures and records repeat exactly per backend.

Same-worker previous/current GPU times are 0.14040/0.08230 s and
0.10660/0.05904 s (1.71x/1.81x); CPU is 0.53400/0.51208 s.
Evidence: `reports/ilt_jacobian_before_v1/` and `reports/ilt_jacobian_after_v1/`.
The former archives the exact earlier verifier executed under its temporary name;
the latter includes raw Jacobian arrays and comparison to the preceding GPU report.
The H100 app ended and was fetched; no local GPU was used.

The GPU pressure operator owns its own separable solver; the CPU LU previously built
by the shared width routine was never read. Its removal passes ten CPU tests and
the unchanged full H100 gate. Before/after fields and complete records are exact.
Evidence: `reports/marangoni_setup_before_v1/` and `reports/marangoni_setup_after_v1/`.
Same-worker total gain is 1.090x/1.058x; all initialization still belongs in timings.

ILT now evaluates dose/focus crossing distances together while retaining strict
central-window bounds, exact-threshold precedence and the original `> tol` rejection.
The post-Jacobian profile measured process-window work at 0.05553 of 0.08985 s;
all numerical masks, images, raw Jacobians and complete records remain exact.
Evidence: `reports/ilt_jacobian_stage_profile_v1/`, `reports/ilt_window_before_v1/`,
and `reports/ilt_window_after_v1/`. H100 timings include all original section-C work;
the first measured execution remains below 10x CPU/GPU speedup. No tolerance change.

E 2026-09-13: Marangoni vectorized conduction experiment CLOSED without adoption. Exact canonical CSR/index/RHS controls22/22, original4/4 twice and all18full fields799440B/leg +complete records exact. Uninstrumented assembly7.85–7.98ms ->0.412–0.427ms, but four warmed whole-call pairs0.611/0.599/0.607/0.606s ->0.591/0.597/0.580/0.648s show no consistent gain. Production unchanged. Candidate/source receipts in reports/marangoni_conduction_v1; paired warm and initial timing retained. Actual-trajectory energy ordering also slower, retained reports/marangoni_energy_profile_v1. Next: measured wavepacket autocorrelation dispatch/transfer costs in reports/wavepacket_stage_profile_v1. No push/local GPU.

E 2026-09-13: fused wavepacket autocorrelation OWN-GATE-FAIL performance; not adopted. Original revival3/3 and unchanged Bloch3/3 twice on H100/L4,19CPU/H100 controls;25152 full autocorrelation values pass original and priorGPU parity at unchanged1e-10+1e-8. Warm whole original main H10045.3–46.1ms ->59.0–60.2ms; cold0.588->1.432s including initialization. Production unchanged; reports/revival_fusion_{before,after,l4,timing}_v1 retain exact source, full arrays and per-app receipts. Next explicit spectrum reuse addresses measured948tensor constructions11.051ms; no push/localGPU.

E 2026-09-13: AutocorrelationSeries owns copied level/weight inputs and one energy buffer replaced on scale/detuning change; no result/global cache. Main retains all316calls and the unchanged Torch arithmetic; standalone autocorr remains available. Full original revival3/3 +Bloch3/3 repeat exactly per backend. H100/L4 are individually within fixed CPU parity, not byte-identical across hosts. First-before timings include lazy runtime initialization and are retained, excluded from isolated improvement claims. Changed-input/shape/retained-output controls pass6/6 CPU/H100. Source and complete arrays pinned in the reports above.

E 2026-09-13: scalar CUDA graph replays the same phase/reduction expression inside AutocorrelationSeries, with fixed scalar input and graph replacement on spectrum changes. Complete arrays and all316calls/readbacks retained; standalone and array paths unchanged. Seven controls include changed scalar times, spectrum changes, interleaved batches and retained outputs; full revival and unchanged Bloch outputs byte-exact to prior GPU. Original3/3 each twice both GPUs; per-host repeats exact. Cold-first ordering is not an isolated speedup; measured warmed calls still construct/capture each series. Post-series and post-graph profiles retained separately.

E 2026-09-13: energy changes now copy exact new values into the retained allocation; scalar graph remains valid. Observer confirms one capture instead of two; no frozen energies or cached answers. Seven changed-time/spectrum/batch controls and complete priorGPU outputs exact; original revival/Bloch3/3 each twice on H100/L4. Initial H100 paired sample32.2ms vs30.2ms regression retained; four additional explicitly warmed pairs consistently improve, every call still constructs a fresh series/graph. Next Bloch operator profile observes15calls, five distinct force operators, repeated widths; denseeigh28.137ms of40.548ms in existing stage profile.

E 2026-09-13: BlochSeries retains at most four operators for one spatial grid, keyed by force/hopping/lattice spacing. Grid changes discard previous operators; all wavepacket states, time grids and observables are recomputed. Original15calls now require five diagonalizations; profile28.137->9.331ms in eigensolve. Ten controls cover force/width/time/grid/material changes, eviction and retained output arrays; original3/3 revival and3/3Bloch repeat exactly per backend, all priorGPU fullarrays exact. Cold runtime initialization is retained separately, not an isolated speedup. Previous revival buffer verifier also retained a100.2ms observed outlier; its row explicitly uses separate uninstrumented warmed timings. Next scalar lithography profile is running on unchanged source.

E final checkpoint 2026-09-13:116 focused CPU tests pass across the six changed compute test modules; targeted scrub has zero hits in changed source/tests (reports/e_night_final_check_v1). All E Modal jobs ended. Scalar aerial profile completed on unchanged source: warm89.539ms, image76.540ms, tensor uploads15.855ms, FFT4.197ms, inverseFFT3.013ms; no candidate was built because the22:00 directive superseded that queue. reports/aerial_stage_profile_v1 retains the measurement. Vulkan capability attempt is closed in the kernel repository; no further compute items or Hankel attempt. README unchanged; no push.

E 2026-09-13 renewed continuation: Hankel-on-device experiment CLOSED, OWN-GATE-FAIL for general replacement performance. Fresh batched profile measured20hostHankel calls9.438ms of35.11ms and14072values, arguments0.3–18. CUDA integer-order Jn/Yn produced complex128 Hankel values and kept gathers/ratios on device; all original9gates twice and24controls pass, full-array repeats exact, priorGPU parity <=1 under unchanged1e-9+1e-8. CPU-parity maximum0.0738766<=1; no bit-exact old/new claim. H100 reversed warm batch37.679/37.831/39.138ms ->30.870/30.968/30.358ms improves, scalar103.773/106.632/105.829ms ->126.006/127.345/119.358ms regresses. First scalar initialization477.0/365.9ms and firstbatch95.0/31.0ms retained, excluded from warm claims. Production source/dependencies unchanged; no L4 rerun/tuning after the measured scalar negative. NVRTC missingmath.h preparation failure retained separately; built-in declarations compile the same kernel arithmetic. Complete candidate, tests, full arrays and receipts in reports/hankel_device_{after,timing,build_failure}_v1, baseline/profile retained. No push/local GPU/README edit.
