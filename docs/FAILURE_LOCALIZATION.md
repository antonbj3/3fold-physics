# Failure-site localization

`chains/failure_localization_v1.py` provides deterministic subset delta debugging
and contiguous-interval bisection. An oracle returns PASS, FAIL or INVALID;
undefined subsets cannot be treated as passing numerical evidence. The subset
result is deletion-minimal: removing any remaining item no longer reproduces the
failure. It is not guaranteed to be globally smallest or the unique physical cause.
The full evaluation trace is retained. Interval bisection records both halves and
stops when neither half independently fails, even if the parent still fails.

The separate real-data observer invokes unchanged KK reconstruction/integration and
boiling flux functions. It also runs both original entrypoints and preserves their
failed exits; the boiling observer must match all four printed baseline gates.
Required real datasets are explicit: missing files refuse instead of silently
substituting synthetic data. Full input, intermediate flux/susceptibility and output
array hashes, source hashes and dataset-file hashes are retained in the report.

```sh
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python src/physics_engine/chains/physics_failure_sites_v1.py
```

## KK: no single measurement site

The five original residuals are1.3026,1.2994,26.1635,3.3174,4.9353.
Their median3.3174 exceeds the unchanged0.5571 threshold, half the original
first-file phase-scrambled control1.1142. Every singleton and every leave-one-out
aggregate still fails against that same control. The deterministic minimizer
returns one singleton witness, but all five can serve as a witness: there is no
unique faulty measurement established here.

The intervention changes only aggregate membership. Each original full frequency
grid, lock-in reconstruction and KK integral is retained. Deleting frequency samples
would also change the nonlocal quadrature and is deliberately not claimed to isolate
an operation defect. These observations do not distinguish finite-band truncation,
measurement assumptions, phase inconsistency or an integration defect, and do not
establish acausality.

## Boiling: ordering and the actual G3 predicate

G1 has exactly one failing pair among the tested pairs:105 and CHF. Mean superheat
rises41.9422 to72.1771 degrees while flux falls916.8745 to521.4390 kW/m2, a change
of-395.4355 kW/m2. This pair is a deletion-minimal witness for the unchanged
monotonicity test; either singleton makes the two-level gate undefined.

G3 has the deletion-minimal pair0 and CHF. The original numeric-tag ordering places
CHF before15 and105. The actual G3 code checks negative unpowered superheat and the
standard deviation of the first powered record. CHF has standard deviation627.7773
against mean521.4390 kW/m2, ratio1.2039323 versus the unchanged0.05 limit.
It does not test unpowered flux near zero: the unpowered mean remains-41.0326 kW/m2.
The observer records this predicate/label mismatch without modifying the baseline.

Bisection of the562176-sample CHF record retains interval[83448,84546), mean1755.9121
and standard deviation98.7151 kW/m2. It fails the same5% predicate. Both549-sample
halves pass individually: their means1841.6372 and1670.1870 differ while their
standard deviations are41.8829 and55.1143. Thus this witness includes variation
between the halves; it does not establish a single isolated bad sample. Other failing sibling
intervals are retained and no unique time site or physical mechanism is claimed.

All five observer gates pass in two byte-identical complete observations. The
original modules remain OWN-GATE-FAIL: KK fails, boiling G1/G3/G4 fail and G2 passes.
The original synthetic boiling control still passes all four gates. Four tests
cover interacting witnesses, invalid subsets, nonlocal interval failures and the
actual G3 predicate. This work localizes the failed checks; it does not repair or
promote either physical model.

## Follow-up: input semantics

The later source-contract audit supersedes a physical interpretation of the KK
subset witnesses. See `KK_INPUT_CONTRACT.md`. Numerical witnesses remain reproducible;
the separate checked adapter refuses these inputs before invoking the KK evaluator.
