"""Illustrative moving heat -> residual stress -> stabilized-cycle fatigue chain.

Every constitutive parameter is declared; no experimental life certificate.
Full-array trajectories are hashed after every independent run. Stage checks
and composition bounds are fixed in docs/RUNNING.md before execution.
"""
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from physics_engine.process.moving_heat_history_v1 import heat_rise
from physics_engine.process import thermal_fiber_bundle_v1 as mechanics
from physics_engine.fatigue import stress_life_v1 as fatigue
from process_chain_seam_probe import ROOT, PARAMS
from process_heat_horizon_probe import observe as heat_controls

NFIBER=41


def digest(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def thermal_field(samples):
    x=np.linspace(.002,-.004,samples)[:,None]
    y=np.linspace(-.0004,.0004,NFIBER)[None,:]
    # Chunk only the outer coordinates; each integral has its own unchanged
    # horizon and quadrature. No reduction spans chunk boundaries.
    return np.concatenate([heat_rise(x[i:i+16],y,np.asarray(1e-5),**PARAMS) for i in range(0,samples,16)])


def temperatures(effective_rise,cooling):
    threshold=mechanics.MELTING-mechanics.AMBIENT
    sensible=np.where(effective_rise<=threshold,effective_rise,
                      np.maximum(threshold,effective_rise-270000/PARAMS['cp']))
    tail=sensible[-1][None,:]*np.linspace(1,0,cooling)[1:,None]
    return mechanics.AMBIENT+np.concatenate((np.zeros((1,NFIBER)),sensible,tail))


def run_state(field,power=1.0,yield_scale=1.0,amplitude=100e6,cooling=257,relief=False):
    temp=temperatures(field*power,cooling)
    process,pdiag=mechanics.history(temp,yield_scale)
    residual=process[-1,:NFIBER]
    plastic=np.zeros(NFIBER) if relief else process[-1,NFIBER:2*NFIBER]
    loads=np.tile([amplitude,-amplitude],20)
    service,sdiag=mechanics.history(np.full((len(loads),NFIBER),mechanics.AMBIENT),yield_scale,plastic,loads)
    last=service[-2:,:NFIBER]
    low,high=last.min(axis=0),last.max(axis=0)
    plastic_increment=float(np.max(np.abs(np.diff(service[-3:,NFIBER:2*NFIBER],axis=0))))
    cycle_difference=float(np.max(np.abs(service[-2:,:NFIBER]-service[-4:-2,:NFIBER])))
    diag=dict(force_error=max(pdiag['force_error'],sdiag['force_error']),
              yield_excess=max(pdiag['yield_excess'],sdiag['yield_excess']),
              dissipation_min=min(pdiag['dissipation_min'],sdiag['dissipation_min']),
              liquid_stress=max(pdiag['liquid_stress'],sdiag['liquid_stress']),
              plastic_increment=plastic_increment,cycle_difference=cycle_difference,
              finite=bool(np.isfinite(process).all() and np.isfinite(service).all()))
    gates=dict(force=diag['force_error']<=1e-10,yield_bound=diag['yield_excess']<=1e-12,
               dissipation=diag['dissipation_min']>=-1e-12,liquid=diag['liquid_stress']==0,
               elastic_service=plastic_increment<=1e-12,stable_service=cycle_difference<=1e-10*mechanics.YIELD,
               finite=diag['finite'])
    return dict(low=low,high=high,residual=residual,gates=gates,diagnostics=diag,
                peak_temperature=float(temp.max()),molten_fibers=int(np.sum(temp.max(axis=0)>=mechanics.MELTING)),
                hashes=dict(temperature=digest(temp),process=digest(process),service=digest(service)),
                array_shapes=dict(temperature=list(temp.shape),process=list(process.shape),service=list(service.shape)))


def evaluation(state,strength_scale=1.0):
    damage=fatigue.damage_per_cycle(state['low'],state['high'],strength=1.5e9*strength_scale)
    return dict(aggregate_damage=float(damage.mean()),max_damage=float(damage.max()),
                damage_sha256=digest(damage),state_hashes=state['hashes'],gates=state['gates'])


def worker():
    h=heat_controls(); m=mechanics.controls(); f=fatigue.controls()
    field=thermal_field(481)
    cache={}
    def state(p=1.0,y=1.0):
        key=(p,y)
        if key not in cache:cache[key]=run_state(field,p,y)
        return cache[key]
    base=state(); refined=run_state(thermal_field(961),cooling=513)
    nominal=evaluation(base)
    fine=evaluation(refined)
    zero=run_state(field,power=0)
    no_load=run_state(field,amplitude=0)
    relieved=run_state(field,relief=True)
    doubled=run_state(field,amplitude=200e6)
    null_evaluations={k:evaluation(v) for k,v in dict(no_heat=zero,no_load=no_load,relieved=relieved,doubled=doubled).items()}
    nulls=dict(zero_residual=float(np.max(np.abs(zero['residual']))),
               zero_load_damage=evaluation(no_load)['aggregate_damage'],
               relieved_damage=evaluation(relieved)['aggregate_damage'],
               doubled_damage=evaluation(doubled)['aggregate_damage'])
    refinement=dict(residual_relative=float(np.max(np.abs(base['residual']-refined['residual']))/mechanics.YIELD),
                    damage_relative=abs(nominal['aggregate_damage']-fine['aggregate_damage'])/fine['aggregate_damage'])
    draws=np.random.default_rng(20260913).uniform(-1,1,(64,3))
    factors=1+draws*np.array([.05,.10,.05])
    subsets=dict(thermal=(0,),mechanical=(1,),fatigue=(2,),joint=(0,1,2),
                 without_thermal=(1,2),without_mechanical=(0,2),without_fatigue=(0,1))
    results={}
    for label,indices in subsets.items():
        rows=[]
        for factor in factors:
            p,y,s=[float(factor[i]) if i in indices else 1.0 for i in range(3)]
            rows.append(evaluation(state(p,y),s))
        results[label]=rows
        print('GROUP '+label+' complete',flush=True)
    spread={k:float(np.std([v['aggregate_damage'] for v in values])) for k,values in results.items()}
    linear=sum(spread[k] for k in ('thermal','mechanical','fatigue'))
    joint=spread['joint']
    reductions={k:joint-spread['without_'+k] for k in ('thermal','mechanical','fatigue')}
    dominant=max(reductions,key=reductions.get)
    # Perturbations must reach an actual intermediate or output field.
    ptest=state(1.05,1.0);ytest=state(1.0,1.1)
    causal=dict(thermal=ptest['hashes']['temperature']!=base['hashes']['temperature'],
                mechanical=ytest['hashes']['process']!=base['hashes']['process'],
                fatigue=evaluation(base,1.05)['damage_sha256']!=nominal['damage_sha256'])
    all_states=list(cache.values())+[refined,zero,no_load,relieved,doubled]
    gates=dict(heat_stage=all(h['gates'].values()),mechanical_controls=all(m['gates'].values()),fatigue_controls=all(f['gates'].values()),
               all_state_gates=all(all(s['gates'].values()) for s in all_states),
               actual_melting=base['molten_fibers']>0,
               residual_refinement=refinement['residual_relative']<.01,damage_refinement=refinement['damage_relative']<.02,
               no_heat=nulls['zero_residual']<=1e-10*mechanics.YIELD,no_load=nulls['zero_load_damage']==0,
               stress_relief=nulls['relieved_damage']<nominal['aggregate_damage'],
               increased_load=nulls['doubled_damage']>nominal['aggregate_damage'],
               causal_stages=all(causal.values()),sublinear_composition=joint<=linear)
    report=dict(scope='Illustrative constitutive chain; no calibrated material lifetime or experimental map prediction',
                gates=gates,stage_controls=dict(heat=h,mechanical=m,fatigue=f),nominal=nominal,
                nominal_state={k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in base.items()},
                nulls=nulls,null_evaluations=null_evaluations,refined=fine,refinement=refinement,causal=causal,draw_factors=factors.tolist(),draws=results,
                composition=dict(spread=spread,linear_sum=linear,joint=joint,ratio=joint/linear,
                                 leave_one_out_reduction=reductions,dominant_stage=dominant),
                state_diagnostics=[s['diagnostics'] for s in all_states],
                state_hashes=[s['hashes'] for s in all_states],
                state_array_shapes=[s['array_shapes'] for s in all_states],
                observed_state_bytes=sum(sum(int(np.prod(shape))*8 for shape in s['array_shapes'].values()) for s in all_states))
    return report


def main():
    if len(sys.argv)==3 and sys.argv[1]=='--worker':
        report=worker()
        Path(sys.argv[2]).write_text(json.dumps(report,indent=2)+'\n')
        return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker REPORT')
    legs=[]
    with tempfile.TemporaryDirectory(prefix='process-chain-') as tmp:
        for i in range(2):
            path=Path(tmp)/f'leg{i}.json'
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',str(path)],cwd=ROOT,check=True)
            legs.append(json.loads(path.read_text()))
            print('LEG '+str(i)+' '+json.dumps(legs[-1]['gates']),flush=True)
    gates=dict(legs[0]['gates'],exact_repeats=legs[0]==legs[1])
    report=dict(legs=legs,gates=gates)
    (ROOT/'reports/process_chain_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(gates=gates,composition=legs[0]['composition'],refinement=legs[0]['refinement'],nulls=legs[0]['nulls'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
