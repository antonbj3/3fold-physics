"""evidence_emit — write a durable, independently checkable record for a computed claim.

emit(name, metrics, gates, provenance, cross_checks, arrays=None) writes
  artifacts/<name>.json  : {name, provenance, metrics (floats), gates, cross_checks}
  artifacts/<name>.npz   : the raw arrays, if any are passed (loadable and plottable by anyone)
and returns the JSON path. `provenance` records where the input data came from (file / scale / units);
`cross_checks` are AUTOMATED checks against an independent reference not used to produce the result
(computed-vs-known-reference, null / planted-fault on known-bad input, or over-determination by two or
more independent paths agreeing within sigma). Output directory: artifacts/ next to this package.
"""
import json, os
import numpy as np

EVID = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts"))


def emit(name, metrics, gates, provenance, cross_checks, arrays=None):
    os.makedirs(EVID, exist_ok=True)
    rec = {
        "name": name,
        "provenance": provenance,                       # {source_file, mesh/scale, units, ...} — where the data came from
        "metrics": {k: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else v) for k, v in metrics.items()},
        "gates": gates,                                 # {G0..Gn: bool, verdict: "PASS"/"REFUSE", why: "..."}
        "cross_checks": cross_checks,                   # automated checks vs INDEPENDENT ground truth (known-ref / null / over-determination)
    }
    jp = os.path.join(EVID, f"{name}.json")
    with open(jp, "w") as f:
        json.dump(rec, f, indent=2)
    if arrays:
        np.savez_compressed(os.path.join(EVID, f"{name}.npz"), **arrays)
    print(f"[EMIT] artifacts/{name}.json" + (f" + {name}.npz ({list(arrays)})" if arrays else "") +
          f" | gates {gates.get('verdict','?')} | {len(cross_checks)} cross-checks")
    return jp
