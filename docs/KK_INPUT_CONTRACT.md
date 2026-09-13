# KK input contract correction

The dataset's angle labels denote propagation directions. The associated paper
defines transmission through an intensity ratio. These are incompatible with the
legacy adapter's lock-in-quadrature interpretation. Sources: [Dryad dataset](https://datadryad.org/dataset/doi:10.5061/dryad.dv41ns1wc)
and [article, section2, figures3/4 and equation2.2](https://pmc.ncbi.nlm.nih.gov/articles/PMC7776964/).

The resulting software conclusion is an input-type error before quadrature: the
legacy expression `(T0-T180)/2 + i*(T90-T270)/2` does not supply the complex response
required by that KK test. Its numerical median3.3174 and gate failure remain
reproducible, but cannot establish a material causality violation, a lock-in phase
fault, or a numerical integration defect. The earlier subset localization only
located failures within that incorrectly interpreted pipeline.

`wave_optics/kk_input_contract_v1.py` is a separate checked entrypoint. It refuses
the known dataset DOI, including DOI-URL notation, even if the caller relabels the
metadata as complex-response data. It also rejects directional quantities, missing
phase, varying configurations, missing frequency units, real-only response arrays,
malformed shapes, nonfinite values and non-increasing frequency grids.

Accepted inputs explicitly declare a complex linear response at one fixed
configuration and a frequency unit of Hz or rad/s. The unchanged KK evaluator is
called only after admission; Hz values are converted to angular frequency. These
are declared semantics, not independently authenticated measurement provenance or
calibration. An arbitrary caller can mislabel anonymous data; this interface does
not solve that general trust problem.

`docs/KK_DATA_CONTRACT.json` records primary-source locations and pins the five
existing local CSV copies. The file inventory and SHA-256 values must match before
the real-data observer runs. These are local-copy pins, not a fresh comparison to
upstream download bytes. The CSV's `Freq` header does not state a unit, so the old
200-800 Hz claim is withdrawn. The observer records raw frequency values without
inventing a unit or silently rescaling the dataset.

Two complex candidates with identical intensities but different phases provide
an algebraic ambiguity control. They are not proposed physical reconstructions.
An analytic damped-oscillator complex-response control reproduces the original
residual exactly and passes its existing gate. No original source or tolerance
is changed.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  python src/physics_engine/wave_optics/kk_input_contract_v1.py
```

Five observer gates pass; eleven refusal/unit/inventory controls pass. The complete
observer report repeats byte-identically in independent processes. Evidence is in
`reports/kk_input_contract_v1.json` and its repeat receipt. The legacy metamaterial
module remains OWN-GATE-FAIL; this new VERIFIED-FRESH row certifies refusal and the
synthetic positive control, not empirical KK validation of the dataset.
