# Data-defined numerical chains

Three JSON specifications live in `src/physics_engine/chains/specs/`:

- `litho_from_spec.json`: the existing aerial/resist/etch composition.
- `process_from_spec.json`: the existing heat/stress/fatigue composition.
- `litho_pore_from_spec.json`: a fourth synthetic pore stage, assembled entirely
  from existing operations by a new spec. This is an uncalibrated stress test.

Run from the repository root with the repository dependencies installed:

```sh
export PYTHONPATH=src
export CUDA_VISIBLE_DEVICES=''
python src/physics_engine/chains/chain_spec_v1.py \
  src/physics_engine/chains/specs/litho_from_spec.json
```

The report goes to `reports/chain_data/litho_from_spec.json`. It includes nominal
outputs, hashes of every returned intermediate, paired uncertainty observations,
joint/linear spread ratios, dominant isolated stages and declared null checks.
A ratio above one is retained in the report. It is not converted into a claim of
sub-linearity, and schema validity never supplies an empirical certificate.

To regenerate the standalone entrypoint:

```sh
python src/physics_engine/chains/chain_spec_v1.py \
  src/physics_engine/chains/specs/litho_from_spec.json \
  src/physics_engine/chains/generated/litho_from_spec.py
python src/physics_engine/chains/generated/litho_from_spec.py
```

A specification contains `schema`, `name`, `scope`, `parameters`, `nodes`,
`output`, `uncertainty` and `nulls`. Each node names an operation from the closed
`OPS` registry and supplies keyword arguments. Bindings use `{"param":"dose"}`
for a declared parameter or `{"ref":"image.0"}` for an earlier node's output.
Numeric path components select tuple/list items; other components select mapping
keys. `skip_if_none` propagates absent geometry through downstream stages.
Unsupported operations, forward references and duplicate node identifiers refuse
execution. Specifications contain no executable Python.

Uncertainty coordinates name a stage group and relative parameter excursions.
Multiple parameters in one coordinate share the same draw, allowing the annular
illumination bounds to move together. Every arm uses the same random coordinate
table; stage Monte-Carlo seeds are fixed in these specs to isolate input
uncertainty. This differs from the separate tail observer, which also varies
intrinsic Monte-Carlo realizations between draws. Their ratios are not directly
interchangeable. At least 64 draws are required.

Null cases declare parameter overrides, an output metric and an equality or
nominal-direction predicate. The process composition also retains every returned
mechanical state gate across its uncertainty observations. Existing numerical
and empirical scope limitations remain attached to the stage implementations.

The complete regeneration certificate runs each generated chain in two separate
processes and checks the two existing chains against direct frozen functions at
nominal, null and individual uncertainty-coordinate inputs:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python src/physics_engine/chains/chain_spec_certify_v1.py
```

See `docs/RUNNING.md` for the measured gates and retained evidence.
