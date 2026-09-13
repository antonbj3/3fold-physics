"""Cross-method checks of the separate downstream-aware heat integral."""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_engine.process.moving_heat_history_v1 import heat_rise
from physics_engine.process.lpbf_meltpool_render_match import dT_rosenthal
from process_chain_seam_probe import PARAMS, ROOT


def observe():
    x=np.linspace(.002,-.004,121)[:,None]
    y=np.linspace(-.0004,.0004,21)[None,:]
    z=np.asarray(1e-5)
    a=heat_rise(x,y,z,**PARAMS,n_u=1500)
    b=heat_rise(x,y,z,**PARAMS,n_u=3000)
    wide=heat_rise(x,y,z,**PARAMS,n_u=6000,horizon_scale=2)
    point=heat_rise(x,y,z,**dict(PARAMS,sigma_b=1e-7))
    closed=dT_rosenthal(x,y,z,**{k:v for k,v in PARAMS.items() if k!='sigma_b'})
    zero=heat_rise(x,y,z,**dict(PARAMS,P=0.0))
    double=heat_rise(x,y,z,**dict(PARAMS,P=400.0))
    arrays=dict(coarse=a,fine=b,wide=wide,point=point,closed=closed,zero=zero,double=double)
    numbers=dict(peak=float(b.max()),final_peak=float(b[-1].max()),point_relative=float(np.linalg.norm(point-closed)/np.linalg.norm(closed)),
                 refinement=float(np.max(np.abs(a-b))/b.max()),tail=float(np.max(np.abs(wide-b))/b.max()),double_relative=float(np.max(np.abs(double-2*b))/b.max()))
    gates=dict(point_limit=numbers['point_relative']<.02,quadrature=numbers['refinement']<.01,tail=numbers['tail']<1e-6,
               zero=bool((zero==0).all()),linearity=numbers['double_relative']<=1e-12,finite_nonnegative=bool(all(np.isfinite(a).all() and (a>=0).all() for a in arrays.values())))
    return dict(numbers=numbers,gates=gates,hashes={k:hashlib.sha256(a.tobytes()).hexdigest() for k,a in arrays.items()})


def main():
    a,b=observe(),observe()
    gates=dict(a['gates'],exact_repeats=a==b)
    report=dict(legs=[a,b],gates=gates)
    (ROOT/'reports/process_heat_horizon_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
