# Boiling LVM channel preservation

The four retained temperature files declare six channels and label seven numeric
columns: time and Temperature_0 through Temperature_5. The frozen loader retains
only the first six columns, omitting Temperature_5. This conclusion follows from
the local file headers and loader code; no physical sensor identity is inferred.

The separate `electrochem/boiling_lvm_channels_v1.py` accepts the observed
single-heading tab-separated layout with explicit Celsius units. It preserves all
seven numeric columns and refuses incomplete columns, unexpected labels, unsupported
metadata, nonfinite samples and non-increasing time. It exports only technical
metadata; acquisition operator names and absolute timestamps are omitted.

Every full parsed array is checked against NumPy's independent text reader. Its
first six columns must match the frozen parser bit for bit. The original
`flux_superheat` function is then applied without changing its arithmetic or column
mapping: both flux and superheat arrays must exactly match the old result. The
additional channel is preserved without assigning it a physical role.

The [dataset article](https://pmc.ncbi.nlm.nih.gov/articles/PMC11239449/) identifies
`Heat_flux.m` as the accompanying analysis code. That file is not present locally;
the dataset API returned HTTP403 during this audit. Therefore the correspondence
between the frozen mapping, sensor depths and published surface-temperature
reconstruction remains unverified. This result does not validate a revised heat
flux, calibration, CHF gate or boiling regime assignment.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  python src/physics_engine/electrochem/boiling_lvm_channels_v1.py
```

Four observer gates pass on all four real files, with complete independent reports
byte-identical. Seven tests cover channel preservation and malformed inputs. Full
file/channel/flux hashes and the unresolved source comparison are retained under
`reports/boiling_lvm_channels_v1.json`; the repeat receipt is separate. Original
boiling gate failures and source code remain unchanged. This reader supports the
observed layout, not every possible LabVIEW format or segmented acquisition.
