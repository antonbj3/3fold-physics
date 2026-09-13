"""Original VOF main/gates and full-trajectory-end parity at all original grids.

Preregistered complete-field budget: atol=1e-7, rtol=1e-6. Every original gate
remains unchanged. Repeated requests within one main invocation reuse the same
fresh result (the original main repeats its null and half-turn diagnostics).
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
import warp as wp
from physics_engine.litho import vof_zalesak_cert as cpu
from physics_engine.litho import vof_zalesak_cert_gpu as gpu
ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-7,1e-6


def cpu_field(n,scheme,revs,cfl):
    fields=[]
    def observer(frame,event,arg):
        if frame.f_code!=cpu.rotate.__code__:return None
        if event=='return':fields.append(frame.f_locals['F'].copy())
        return observer
    prev=sys.gettrace()
    try:
        sys.settrace(observer);result=cpu.rotate(n,scheme,revs,cfl)
    finally:sys.settrace(prev)
    return result,fields[0]


def run_main(device,reference=False):
    records={};fields={};times={};gates={}
    def rotate(n,scheme='geom',revs=1.,cfl=.5):
        key=f'{n}_{scheme}_{revs}_{cfl}'
        if key not in records:
            start=time.perf_counter()
            if reference:result,F=cpu_field(n,scheme,revs,cfl)
            else:result,F=gpu.rotate(n,scheme,revs,cfl,device=device,return_field=True)
            times[key]=time.perf_counter()-start
            records[key]=result;fields[key]=F
        return records[key]
    scope=dict(vars(cpu));scope['rotate']=rotate
    fn=types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)
    # CPU field tracing has overhead: separate unobserved timing is recorded below.
    with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    payload=json.loads((ROOT/'artifacts/vof_zalesak_cert.json').read_text())
    return records,fields,times,payload,rc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/vof_gpu_v1');args=p.parse_args()
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    wp.init()
    if args.device=='cuda' and not wp.is_cuda_available():raise SystemExit('CUDA required')
    gpu.rotate(8,revs=.01,device=args.device)
    runs=[];capture=[]
    for repeat in range(2):
        cr,ca,ct,cp,rc=run_main(args.device,True)
        gr,ga,gt,gp,rg=run_main(args.device,False)
        (out/f'cpu_gates_{repeat}.json').write_text(json.dumps(cp,indent=2)+'\n')
        (out/f'gpu_gates_{repeat}.json').write_text(json.dumps(gp,indent=2)+'\n')
        arrays={**{'cpu_'+k:v for k,v in ca.items()},**{'gpu_'+k:v for k,v in ga.items()}}
        np.savez_compressed(out/f'fields_{repeat}.npz',**arrays);capture.append(arrays)
        metrics=[]
        for key in ca:
            error=float(np.max(abs(ga[key]-ca[key])/(ATOL+RTOL*abs(ca[key]))))
            # Uninstrumented original CPU timing; verify its full scalar result
            # equals the instrumented invocation before accepting the timing.
            n,scheme,revs,cfl=key.split('_');start=time.perf_counter()
            timed=cpu.rotate(int(n),scheme,float(revs),float(cfl));tc=time.perf_counter()-start
            if timed!=cr[key]:raise RuntimeError('observer changed CPU result')
            metrics.append(dict(case=key,cpu_s=tc,gpu_s=gt[key],cpu_observed_s=ct[key],
                parity=error,max_abs_error=float(np.max(abs(ga[key]-ca[key]))),
                cpu=cr[key],gpu=gr[key]))
        runs.append(dict(cases=metrics,cpu_gates=cp['gates'],gpu_gates=gp['gates'],
            all_pass=rc==rg==0 and cp['ok'] and gp['ok'] and all(m['parity']<=1 for m in metrics)))
    exact=all(np.array_equal(capture[0][k],capture[1][k]) for k in capture[0])
    files=[Path(cpu.__file__),Path(gpu.__file__),Path(__file__)]
    report=dict(device=str(wp.get_device(args.device)),atol=ATOL,rtol=RTOL,runs=runs,
        full_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
