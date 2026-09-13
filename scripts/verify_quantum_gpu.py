"""Original Ising/FD-Schrodinger gates, full histories and spectral parity.

Ising spin values, measurement series and RNG exit state must match exactly.
Schrodinger eigenvalues/vectors use atol=1e-8, rtol=1e-6; each real eigenvector
may differ by its unobservable sign, recorded explicitly without altering capture.
"""
import argparse
import contextlib
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
import torch
import warp as wp
from physics_engine.quantum import ising_2d as ising
from physics_engine.quantum import ising_2d_gpu as ig
from physics_engine.quantum import schrodinger_1d as sch
from physics_engine.quantum import schrodinger_1d_gpu as sg
ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-8,1e-6


def suite(name,device,reference,series=False):
    module=ising if name=='ising' else sch
    scope=dict(vars(module));fields={};records=[];gates={}
    full_fn=ig.SimulationSeries(device,return_history=True) if series else None
    scalar_fn=ig.SimulationSeries(device) if series else None
    def simulate(L,T,rng,eq=400,meas=600):
        scalar_rng=copy.deepcopy(rng) if not reference else None
        index=len(records);start=time.perf_counter()
        if reference:
            histories=[];calls=0
            def sweep(*a):
                nonlocal calls
                value=ising.sweep(*a);calls+=1
                if calls>2*eq and calls%2==0:histories.append(value.copy())
                return value
            local=dict(vars(ising));local['sweep']=sweep
            fn=types.FunctionType(ising.simulate.__code__,local,ising.simulate.__name__,ising.simulate.__defaults__)
            result=fn(L,T,rng,eq,meas);states=np.array(histories)
            M=np.abs(states.mean(axis=(1,2)))
            E=-(states*(np.roll(states,1,axis=1)+np.roll(states,1,axis=2))).mean(axis=(1,2))
        else:result,states,M,E=full_fn(L,T,rng,eq,meas) if series else ig.simulate(L,T,rng,eq,meas,device=device,return_history=True)
        seconds=time.perf_counter()-start
        fields.update({f'{index}_spins':states,f'{index}_M':M,f'{index}_E':E})
        record=dict(T=float(T),seconds=seconds,result=result,rng_state=rng.bit_generator.state)
        if not reference:
            start=time.perf_counter()
            scalar=scalar_fn(L,T,scalar_rng,eq,meas) if series else ig.simulate(L,T,scalar_rng,eq,meas,device=device)
            record.update(scalar_seconds=time.perf_counter()-start,
                          scalar_result=scalar,
                          scalar_result_rng_exact=scalar==result and scalar_rng.bit_generator.state==rng.bit_generator.state)
        records.append(record)
        return result
    def solve(V,dx,k=6):
        index=len(records)
        if not reference and device=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();E,P=sch.solve(V,dx,k) if reference else sg.solve(V,dx,k,device)
        if not reference and device=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        fields.update({f'{index}_E':E,f'{index}_psi':P,f'{index}_V':V.copy()})
        records.append(dict(dx=float(dx),seconds=seconds))
        return E,P
    scope['simulate' if name=='ising' else 'solve']=simulate if name=='ising' else solve
    fn=types.FunctionType(module.main.__code__,scope,module.main.__name__)
    def observer(frame,event,arg):
        if event=='return' and frame.f_code==module.main.__code__:
            gates.update({key:bool(frame.f_locals[key]) for key in ('ok1','ok2','ok3','ok4')})
    prev=sys.getprofile()
    try:
        sys.setprofile(observer)
        with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    finally:sys.setprofile(prev)
    return fields,records,gates,rc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/quantum_gpu_v1')
    p.add_argument('--module',choices=['ising','schrodinger'],help='run only the selected existing module checks')
    p.add_argument('--ising-series',action='store_true',help='reuse explicit buffers across each original temperature series')
    args=p.parse_args()
    torch.set_num_threads(1);wp.init()
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    ig.simulate(4,2.,np.random.default_rng(1),eq=1,meas=1,device=args.device)
    if args.module!='ising':sg.solve(np.ones(16),.1,device=args.device)
    rows=[]
    for name in ([args.module] if args.module else ('ising','schrodinger')):
        captures=[];runs=[]
        for repeat in range(2):
            ca,cr,cg,rc=suite(name,args.device,True);ga,gr,gg,rg=suite(name,args.device,False,args.ising_series)
            arrays={**{'cpu_'+k:v for k,v in ca.items()},**{'gpu_'+k:v for k,v in ga.items()}}
            captures.append(arrays);np.savez_compressed(out/f'{name}_{repeat}.npz',**arrays)
            if name=='ising':
                parity=all(np.array_equal(ca[k],ga[k]) for k in ca)
                control=all(c['result']==g['result'] and c['rng_state']==g['rng_state'] for c,g in zip(cr,gr))
                scalar_exact=all(g['scalar_result_rng_exact'] for g in gr)
                control=control and scalar_exact
                detail=dict(spins_and_histories_exact=parity,results_rng_exit_exact=control,
                            scalar_result_rng_exact=scalar_exact)
            else:
                errors=[];signs=[];residual=[]
                for index,record in enumerate(cr):
                    P,Q=ca[f'{index}_psi'],ga[f'{index}_psi']
                    sign=np.where(np.sum(P*Q,axis=0)<0,-1.,1.);signs.append(sign.tolist());Q=Q*sign
                    E,G=ca[f'{index}_E'],ga[f'{index}_E'];dx=record['dx'];V=ca[f'{index}_V']
                    errors.extend([float(np.max(abs(P-Q)/(ATOL+RTOL*abs(P)))),float(np.max(abs(E-G)/(ATOL+RTOL*abs(E))))])
                    HQ=(1/dx**2+V)[:,None]*Q
                    HQ[1:]+=-.5/dx**2*Q[:-1];HQ[:-1]+=-.5/dx**2*Q[1:]
                    residual.append(float(np.max(abs(HQ-Q*G))))
                parity=max(errors)<=1;control=max(residual)<1e-6
                detail=dict(normalized_parity=max(errors),eigenvector_signs=signs,stencil_residual_inf=residual)
            runs.append(dict(cpu_gates=cg,gpu_gates=gg,cpu_cases=cr,gpu_cases=gr,parity=detail,
                all_pass=rc==rg==0 and all(cg.values()) and all(gg.values()) and parity and control))
        exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0])
        rows.append(dict(module=name,runs=runs,full_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs)))
    files=[Path(m.__file__) for m in (ising,ig,sch,sg)]+[Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',atol=ATOL,rtol=RTOL,
        ising_series=args.ising_series,rows=rows,all_pass=all(r['all_pass'] for r in rows),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
