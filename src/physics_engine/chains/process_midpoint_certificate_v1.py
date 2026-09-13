"""Full declared ensemble and null refinement of cell-centered process sampling."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
import process_midpoint_v1 as variant
from process_chain_v1 import ROOT, heat_controls


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [plain(v) for v in value]
    return value


def difference(coarse,fine):
    a=coarse['evaluation']['aggregate_damage'];b=fine['evaluation']['aggregate_damage']
    return dict(damage_relative=abs(a-b)/b if b else (0. if a==0. else float('inf')),
        residual_relative=float(np.max(np.abs(np.array(coarse['residual'])-np.array(fine['residual'])[1::3]))/variant.mechanics.YIELD))


def observe():
    stage_controls=dict(heat=heat_controls(),mechanics=variant.mechanics.controls(),fatigue=variant.fatigue.controls())
    factors=1+np.random.default_rng(20260913).uniform(-1,1,(64,3))*np.array([.05,.10,.05])
    subsets=dict(thermal=(0,),mechanical=(1,),fatigue=(2,),joint=(0,1,2),without_thermal=(1,2),without_mechanical=(0,2),without_fatigue=(0,1))
    resolutions={};all_gates=[]
    for count in (123,369):
        field=variant.thermal_field(961,count);cache={}
        def state(p=1.,y=1.,amplitude=100e6,relief=False):
            key=(p,y,amplitude,relief)
            if key not in cache:cache[key]=variant.run_state(field,p,y,amplitude,relief=relief)
            return cache[key]
        def row(s,strength=1.):
            return dict(evaluation=variant.evaluation(s,strength),residual=s['residual'].tolist(),
                molten=s['molten_fibers'],diagnostics=s['diagnostics'],state_shapes=s['array_shapes'])
        nominal=row(state());groups={}
        for label,indices in subsets.items():
            rows=[]
            for factor in factors:
                p,y,s=[float(factor[i]) if i in indices else 1. for i in range(3)]
                rows.append(row(state(p,y),s))
            groups[label]=rows
        nulls=dict(no_heat=row(state(p=0.)),no_load=row(state(amplitude=0.)),relieved=row(state(relief=True)),doubled=row(state(amplitude=200e6)))
        causal=dict(thermal=state(1.05)['hashes']['temperature']!=state()['hashes']['temperature'],
                    mechanical=state(y=1.1)['hashes']['process']!=state()['hashes']['process'],
                    fatigue=variant.evaluation(state(),1.05)['damage_sha256']!=nominal['evaluation']['damage_sha256'])
        spreads={label:float(np.std([r['evaluation']['aggregate_damage'] for r in rows])) for label,rows in groups.items()}
        linear=sum(spreads[g] for g in ('thermal','mechanical','fatigue'));joint=spreads['joint']
        reductions={g:joint-spreads['without_'+g] for g in ('thermal','mechanical','fatigue')}
        gates=dict(state_gates=all(all(s['gates'].values()) for s in cache.values()),actual_melting=nominal['molten']>0,
            no_heat=max(abs(v) for v in nulls['no_heat']['residual'])<=1e-10*variant.mechanics.YIELD,
            no_load=nulls['no_load']['evaluation']['aggregate_damage']==0.,
            stress_relief=nulls['relieved']['evaluation']['aggregate_damage']<nominal['evaluation']['aggregate_damage'],
            increased_load=nulls['doubled']['evaluation']['aggregate_damage']>nominal['evaluation']['aggregate_damage'],
            causal_stages=all(causal.values()),sublinear_composition=joint<=linear)
        all_gates.append(gates)
        resolutions[str(count)]=dict(nominal=nominal,groups=groups,nulls=nulls,causal=causal,
            causal_rows=dict(thermal=row(state(1.05)),mechanical=row(state(y=1.1))),
            state_hashes=[s['hashes'] for s in cache.values()],gates=gates,
            composition=dict(spreads=spreads,linear_sum=linear,joint=joint,ratio=joint/linear,dominant=max(reductions,key=reductions.get)),
            field_sha256=variant.digest(field),unique_states=len(cache))
        print('resolution',count,'complete',flush=True)
    a,b=resolutions['123'],resolutions['369'];pairs={}
    pairs['nominal']=difference(a['nominal'],b['nominal'])
    for group in subsets:
        for i,(c,f) in enumerate(zip(a['groups'][group],b['groups'][group])):pairs[f'{group}:{i}']=difference(c,f)
    for name in a['nulls']:pairs['null:'+name]=difference(a['nulls'][name],b['nulls'][name])
    for name in a['causal_rows']:pairs['causal:'+name]=difference(a['causal_rows'][name],b['causal_rows'][name])
    fine_time=variant.run_state(variant.thermal_field(1921,123),cooling=1025)
    fine_eval=variant.evaluation(fine_time)
    time_errors=dict(damage_relative=abs(a['nominal']['evaluation']['aggregate_damage']-fine_eval['aggregate_damage'])/fine_eval['aggregate_damage'],
        residual_relative=float(np.max(np.abs(np.array(a['nominal']['residual'])-fine_time['residual']))/variant.mechanics.YIELD))
    frozen_field=variant.frozen.thermal_field(481)
    expected=variant.frozen.run_state(frozen_field)
    candidate=variant.run_state(frozen_field,cooling=257)
    exact_frozen=plain(expected)==plain(candidate)
    gates=dict(heat_stage=all(stage_controls['heat']['gates'].values()),mechanical_controls=all(stage_controls['mechanics']['gates'].values()),
        fatigue_controls=all(stage_controls['fatigue']['gates'].values()),both_resolution_contracts=all(all(g.values()) for g in all_gates),
        time_damage=time_errors['damage_relative']<.02,time_residual=time_errors['residual_relative']<.01,
        refined_time_state=all(fine_time['gates'].values()),frozen_input_exact=exact_frozen,
        ensemble_damage=all(v['damage_relative']<.02 for v in pairs.values()),
        ensemble_residual=all(v['residual_relative']<.01 for v in pairs.values()))
    return dict(scope='Numerical refinement on declared illustrative ensemble; no calibrated material life',gates=gates,stage_controls=stage_controls,
        factors=factors.tolist(),resolutions=resolutions,refinement=pairs,time_refinement=time_errors,
        time_state_hashes=fine_time['hashes'],time_damage_sha256=fine_eval['damage_sha256'],
        frozen_input_hashes=dict(expected=expected['hashes'],candidate=candidate['hashes']),
        worst={key:max(pairs,key=lambda p:pairs[p][key]) for key in ('damage_relative','residual_relative')})


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        Path(sys.argv[2]).write_text(json.dumps(plain(observe()),sort_keys=True,allow_nan=False));return 0
    legs=[]
    with tempfile.TemporaryDirectory(prefix='midpoint-certificate-') as tmp:
        for i in range(2):
            p=Path(tmp)/f'{i}.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(p)],check=True,timeout=900)
            legs.append(json.loads(p.read_text()));print('leg',i,'complete',flush=True)
    gates=dict(legs[0]['gates'],exact_repeat=legs[0]==legs[1])
    result=dict(legs=legs,gates=gates,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    p=ROOT/'reports/process_midpoint_certificate_v1.json';p.parent.mkdir(exist_ok=True);p.write_text(json.dumps(result,sort_keys=True,allow_nan=False)+'\n')
    print(json.dumps(dict(gates=gates,worst={k:legs[0]['refinement'][name] for k,name in legs[0]['worst'].items()},time=legs[0]['time_refinement'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
