"""Observe the frozen failed corner without changing any integration arithmetic."""
import inspect
import json
from pathlib import Path
import sys

import numpy as np

import battery_chain_v1 as reference


def measure():
    target = reference.simulate.__code__
    lines,start = inspect.getsourcelines(reference.simulate)
    observed_line = next(start+i for i,line in enumerate(lines) if 'digest.update(snapshot.tobytes())' in line)
    lower=0.0;upper=0.0;first=None
    def trace(frame,event,arg):
        nonlocal lower,upper,first
        if frame.f_code is target and event=='line' and frame.f_lineno==observed_line:
            local = frame.f_locals
            values = np.r_[local['soc'],local['state'][:4]]
            lo,hi = float(values.min()),float(values.max())
            lower=min(lower,lo);upper=max(upper,hi-1.0)
            if (lo<0 or hi>1) and first is None:
                first={'step':local['count'],'lower':lo,'upper_excess':max(0.,hi-1),
                       'state':values.tolist()}
        return trace
    previous=sys.gettrace()
    try:
        sys.settrace(trace)
        result=reference.simulate((-0.01,reference.COEFF[-1],1.1))
    finally:
        sys.settrace(previous)
    return {'result':result,'minimum_below_zero':lower,'maximum_above_one':upper,'first_violation':first}


def main():
    a,b=measure(),measure()
    saved=json.loads((reference.ROOT/'reports/battery_chain_v1.json').read_text())['measurement']['corners']
    original=next(r for r in saved if r['parameters']==[-0.01,reference.COEFF[-1],1.1])
    gates={'two_complete_results_identical':a==b,
           'instrumentation_preserves_trajectory':a['result']['trajectory_sha256']==original['trajectory_sha256'],
           'strict_bounds':a['minimum_below_zero']>=0 and a['maximum_above_one']<=0}
    report={'measurement':a,'gates':gates,'scope':'synthetic failed-corner observation; no tolerance relaxation'}
    (reference.ROOT/'reports/battery_chain_inventory_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
