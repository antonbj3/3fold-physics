# Whole-record stationarity and block aggregation

This observer uses every sample from the four retained real temperature records.
The block size is1024, checked against each file's acquisition header. No window
is selected or discarded. Heat-flux arithmetic and channel assignment remain the
frozen module's unverified physical mapping.

A record is admitted for stationary positive-flux statistics only when its full
mean is positive and its full standard deviation is below5% of that mean. This
uses the existing numerical relative-spread limit explicitly for every record.
Individual passing blocks never override a failed full-record criterion.

| Record | Samples / blocks | Full relative spread | Locally admitted blocks | Whole record |
|---|---|---|---|---|
|0|398336 /389|3.503821%|0 /389|Refused: signed mean negative|
|15|390144 /381|1.000832%|381 /381|Admitted|
|105|392192 /383|0.164033%|383 /383|Admitted|
|CHF|562176 /549|120.393233%|547 /549|Refused|

For equal-sized blocks, total population variance is the mean within-block
variance plus the variance of block means. In the CHF record,99.9883631743% of the
variance is between block means. Locally stable blocks therefore do not make this
record stationary. The complete block means, standard deviations, timing endpoints
and admission flags are retained, including the two locally failing blocks.

The decomposition residual is checked against64 times float64 machine epsilon,
scaled by the largest variance term or1. This is a declared arithmetic check, not
a physical tolerance or a change to the5% criterion. A synthetic pair of constant
blocks1 and2 has zero within-block variance and total variance0.25; both blocks
pass individually while the full record fails. This is a direct counterexample to
acceptance by local-window stability alone.

The admitted pair15/105 remains ordered in mean flux versus mean superheat. That
is a conditional numerical statement after explicit whole-record admission, not a
repair of the original all-record G1 gate. The full CHF record and signed null
remain visible and refused. No stationary CHF estimate is obtained by choosing
convenient windows. The sensor-depth mapping, surface reconstruction and physical
regime attribution remain unverified; no film-boiling diagnosis is made here.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 \
  python src/physics_engine/electrochem/boiling_stationarity_v1.py
```

Seven observer gates and six tests pass. Independent complete reports are byte
identical. Full flux/superheat hashes equal the prior channel-preservation capture.
The complete report and repeat receipt are
`reports/boiling_stationarity_v1.json` and
`reports/boiling_stationarity_repeat_v1.json`. Original numerical source, data and
failed physical gates are unchanged.
