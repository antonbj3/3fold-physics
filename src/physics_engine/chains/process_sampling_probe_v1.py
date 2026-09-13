"""Measure endpoint bias versus cell-center quadrature in the frozen chain."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from physics_engine.chains import process_chain_v1 as chain


def observe(kind,count):
    chain.NFIBER=count
    x=np.linspace(.002,-.004,961)[:,None]
    if kind=='endpoint':y=np.linspace(-.0004,.0004,count)
    elif kind=='midpoint':y=-.0004+(np.arange(count)+.5)*(.0008/count)
    else:raise ValueError('Unknown quadrature')
    field=np.concatenate([chain.heat_rise(x[i:i+16],y[None,:],np.asarray(1e-5),**chain.PARAMS) for i in range(0,len(x),16)])
    state=chain.run_state(field,cooling=513)
    return dict(kind=kind,count=count,y=y.tolist(),field_sha256=chain.digest(field),result=chain.evaluation(state),
        residual=state['residual'].tolist(),molten=state['molten_fibers'],diagnostics=state['diagnostics'])


def experiment():
    rows={kind:[observe(kind,n) for n in counts] for kind,counts in
          (('endpoint',(41,81,161)),('midpoint',(41,123,369)))}
    comparisons={}
    for kind,values in rows.items():
        comparisons[kind]=[]
        for coarse,fine in zip(values,values[1:]):
            indices=np.arange(0,fine['count'],2) if kind=='endpoint' else np.arange(1,fine['count'],3)
            coordinate_error=float(np.max(np.abs(np.array(coarse['y'])-np.array(fine['y'])[indices])))
            residual=float(np.max(np.abs(np.array(coarse['residual'])-np.array(fine['residual'])[indices]))/chain.mechanics.YIELD)
            c=coarse['result']['aggregate_damage'];f=fine['result']['aggregate_damage']
            comparisons[kind].append(dict(coarse=coarse['count'],fine=fine['count'],coordinate_error=coordinate_error,
                residual_relative=residual,damage_relative=abs(c-f)/f))
    return dict(rows=rows,comparisons=comparisons)


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        Path(sys.argv[2]).write_text(json.dumps(experiment(),sort_keys=True,allow_nan=False));return 0
    legs=[]
    with tempfile.TemporaryDirectory(prefix='process-sampling-') as tmp:
        for leg in range(2):
            output=Path(tmp)/f'{leg}.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(output)],check=True,timeout=600)
            legs.append(json.loads(output.read_text()));print('leg',leg,'complete',flush=True)
    gates=dict(exact_repeat=legs[0]==legs[1],state_gates=all(all(r['result']['gates'].values()) for rows in legs[0]['rows'].values() for r in rows),
        nested_coordinates=all(c['coordinate_error']<1e-18 for rows in legs[0]['comparisons'].values() for c in rows),
        frozen_failure_reproduced=legs[0]['comparisons']['endpoint'][0]['damage_relative']>.02,
        midpoint_damage=all(c['damage_relative']<.02 for c in legs[0]['comparisons']['midpoint']),
        midpoint_residual=all(c['residual_relative']<.01 for c in legs[0]['comparisons']['midpoint']))
    report=dict(legs=legs,gates=gates,scope='Nominal numerical sampling observer; no ensemble or empirical accuracy certificate',
        status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    p=chain.ROOT/'reports/process_sampling_probe_v1.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(gates=gates,comparisons=legs[0]['comparisons'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
