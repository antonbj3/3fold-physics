"""Audit transverse convergence of the existing illustrative process chain."""
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from physics_engine.chains import process_chain_v1 as chain


def worker(count):
    chain.NFIBER=count
    state=chain.run_state(chain.thermal_field(961),cooling=513)
    result=chain.evaluation(state)
    return dict(fibers=count,result=result,residual=state['residual'].tolist(),
                diagnostics=state['diagnostics'],molten_fibers=state['molten_fibers'],
                state_shapes=state['array_shapes'])


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        print('RESULT '+json.dumps(worker(int(sys.argv[2]))),flush=True)
        return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker COUNT')
    legs=[]
    for _ in range(2):
        rows=[]
        for count in (41,81,161):
            p=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(count)],
                             cwd=chain.ROOT,capture_output=True,text=True,check=True)
            lines=[line[7:] for line in p.stdout.splitlines() if line.startswith('RESULT ')]
            if len(lines)!=1:raise RuntimeError('Expected one worker result')
            rows.append(json.loads(lines[0]))
            print('COUNT '+str(count)+' damage '+str(rows[-1]['result']['aggregate_damage']),flush=True)
        comparisons=[]
        for coarse,fine in zip(rows,rows[1:]):
            # Nested uniform coordinates: every second fine node coincides.
            c=np.array(coarse['residual']);f=np.array(fine['residual'])[::2]
            comparisons.append(dict(coarse=coarse['fibers'],fine=fine['fibers'],
                                    damage_relative=abs(coarse['result']['aggregate_damage']-fine['result']['aggregate_damage'])/fine['result']['aggregate_damage'],
                                    residual_relative=float(np.max(np.abs(c-f))/chain.mechanics.YIELD)))
        legs.append(dict(rows=rows,comparisons=comparisons))
    gates=dict(original_state_gates=all(all(r['result']['gates'].values()) for leg in legs for r in leg['rows']),
               actual_melting=all(r['molten_fibers']>0 for leg in legs for r in leg['rows']),
               damage_refinement=all(c['damage_relative']<.02 for leg in legs for c in leg['comparisons']),
               residual_refinement=all(c['residual_relative']<.01 for leg in legs for c in leg['comparisons']),
               exact_repeats=legs[0]==legs[1])
    report=dict(legs=legs,gates=gates,scope='Transverse convergence only; illustrative material')
    (chain.ROOT/'reports/process_transverse_refinement_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(gates=gates,comparisons=legs[0]['comparisons'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
