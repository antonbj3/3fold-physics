"""Observe frozen thermal histories before constructing process-chain adapters."""
import hashlib
import json
from pathlib import Path
import numpy as np
from physics_engine.process import lpbf_meltpool_render_match as thermal

ROOT = Path(__file__).resolve().parents[3]
PARAMS = dict(P=200.0, v=0.8, eta=0.35, sigma_b=40e-6, k=20.0, rho=8200.0, cp=600.0)


def observe():
    x = np.linspace(0.002, -0.004, 121)[:, None]
    y = np.linspace(-0.0004, 0.0004, 21)[None, :]
    z = np.asarray(1e-5)
    a = thermal.dT_eagar_tsai(x, y, z, **PARAMS, n_u=1500)
    b = thermal.dT_eagar_tsai(x, y, z, **PARAMS, n_u=3000)
    zero = thermal.dT_eagar_tsai(x,y,z,**dict(PARAMS,P=0.0))
    double = thermal.dT_eagar_tsai(x,y,z,**dict(PARAMS,P=400.0))
    point = thermal.dT_eagar_tsai(x,y,z,**dict(PARAMS,sigma_b=1e-7),n_u=3000)
    closed = thermal.dT_rosenthal(x,y,z,**{k:v for k,v in PARAMS.items() if k!='sigma_b'})
    arrays = dict(coarse=a, fine=b, zero=zero, double=double, point=point, closed=closed)
    report = dict(parameters=PARAMS, shape=list(a.shape), peak_rise=float(a.max()),
                  min_peak_across_fibers=float(a.max(axis=0).min()),
                  max_initial_rise=float(a[0].max()), max_final_rise=float(a[-1].max()),
                  melt_threshold=1600-293+270000/600,
                  molten_fibers=int(np.sum(a.max(axis=0)>=1600-293+270000/600)),
                  quadrature_relative=float(np.max(np.abs(a-b))/b.max()),
                  point_relative=float(np.linalg.norm(point-closed)/np.linalg.norm(closed)),
                  double_relative=float(np.max(np.abs(double-2*a))/a.max()),
                  hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in arrays.items()},
                  gates=dict(finite_nonnegative=bool(all(np.isfinite(v).all() and (v>=0).all() for v in arrays.values())),
                             zero_power=bool((zero==0).all()),
                             power_linearity=bool(np.max(np.abs(double-2*a))/a.max()<=1e-12),
                             quadrature=bool(np.max(np.abs(a-b))/b.max()<0.01),
                             point_limit=bool(np.linalg.norm(point-closed)/np.linalg.norm(closed)<0.02)))
    return report


def main():
    a,b=observe(),observe()
    gates=dict(a['gates'], exact_repeats=a==b)
    report=dict(legs=[a,b],gates=gates,scope='Illustrative material; existing empirical stage certificates not transferred')
    (ROOT/'reports').mkdir(exist_ok=True)
    (ROOT/'reports/process_chain_seam_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
