"""Original Marangoni gates, six surface-tension cases and full T/u/w parity.

The full main is reused with per-invocation reuse of repeated identical solves.
Pointwise field budget fixed before CUDA: atol=1e-7, rtol=1e-6. Original stopping
criteria and partial-closure semantics remain unchanged. Timings include all host
conduction/energy work and GPU momentum/projection work; no speed gate is added.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
import torch
from physics_engine.process import marangoni_meltpool_cfd as cpu
from physics_engine.process import marangoni_meltpool_cfd_gpu as gpu
ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-7,1e-6


def suite(device,reference=False):
    scope=dict(vars(cpu)) if reference else gpu.solver_scope(device)
    solve=cpu.width_ratio if reference else scope['width_ratio']
    records={};fields={};times={};iterations={};gates={}
    active=[]
    def observer(frame,event,arg):
        if event=='return' and frame.f_code==cpu.segregated_solve.__code__:
            key=active[-1]
            fields.update({key+'_'+name:frame.f_locals[name].copy() for name in ('T','u','w')})
            iterations[key]=dict(outer=int(frame.f_locals['outer']),last_sub=int(frame.f_locals['ns']))
        if event=='return' and frame.f_code==cpu.main.__code__:
            gates.update(g4=bool(frame.f_locals['g4']),g5=bool(frame.f_locals['g5']),
                nulls=bool(frame.f_locals['res'].nulls_ok),perturbations=bool(frame.f_locals['res'].perts_ok))
    def width(dsdt,nx=100,nz=55,want=('ratio',)):
        key=f'{dsdt}_{nx}_{nz}'
        if key not in records:
            active.append(key)
            if device=='cuda' and not reference:torch.cuda.synchronize()
            start=time.perf_counter();result=solve(dsdt,nx,nz,('all',))
            if device=='cuda' and not reference:torch.cuda.synchronize()
            times[key]=time.perf_counter()-start;records[key]=result;active.pop()
        return records[key] if len(want)>1 or want[0]!='ratio' else records[key]['ratio']
    scope['width_ratio']=width
    fn=types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)
    prev=sys.getprofile()
    try:
        sys.setprofile(observer)
        with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    finally:sys.setprofile(prev)
    return records,fields,times,iterations,gates,rc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/marangoni_gpu_v1');args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    gpu.Pressure(7,5,1.,1.,args.device).solve(np.zeros(35))
    captures=[];runs=[]
    for repeat in range(2):
        cr,ca,ct,ci,cg,rc=suite(args.device,True)
        gr,ga,gt,gi,gg,rg=suite(args.device,False)
        arrays={**{'cpu_'+k:v for k,v in ca.items()},**{'gpu_'+k:v for k,v in ga.items()}}
        np.savez_compressed(out/f'fields_{repeat}.npz',**arrays);captures.append(arrays)
        (out/f'records_{repeat}.json').write_text(json.dumps(dict(cpu=cr,gpu=gr,cpu_iterations=ci,gpu_iterations=gi),indent=2)+'\n')
        err=max(float(np.max(abs(ga[k]-ca[k])/(ATOL+RTOL*abs(ca[k])))) for k in ca)
        scalar=max(abs(gr[k][v]-cr[k][v])/(ATOL+RTOL*abs(cr[k][v])) for k in cr for v in cr[k])
        runs.append(dict(cpu_gates=cg,gpu_gates=gg,cpu_seconds=ct,gpu_seconds=gt,
            field_parity=err,scalar_parity=scalar,iterations_exact=ci==gi,
            all_pass=rc==rg==0 and all(cg.values()) and all(gg.values()) and err<=1 and scalar<=1 and ci==gi))
    exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0])
    files=[Path(cpu.__file__),Path(gpu.__file__),Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',
        atol=ATOL,rtol=RTOL,runs=runs,full_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
