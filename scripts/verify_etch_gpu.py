"""Original etch gates, complete Euler histories and a 256-width batch timing."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import warp as wp
from physics_engine.litho import litho_etch_levelset_ballistic as cpu
from physics_engine.litho import litho_etch_levelset_ballistic_gpu as gpu
from physics_engine.litho import vof_zalesak_cert_gpu as vof
ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-12,1e-10


def own(fn):
    gates={};arrays={}
    def observer(frame,event,arg):
        if frame.f_code!=cpu.main.__code__:return None
        if event=='return':
            gates.update({k:bool(frame.f_locals[k]) for k in ('g1','g2','g3','g4')})
            arrays.update({k:frame.f_locals[k].copy() for k in ('tw','hw','rw','tn','hn','rn')})
        return observer
    prev=sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    finally:sys.settrace(prev)
    return gates,arrays,rc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/etch_gpu_v1');args=p.parse_args()
    wp.init()
    if args.device=='cuda' and not wp.is_cuda_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    gpu.etch_batch([1.,4.],device=args.device)
    runs=[];captures=[]
    for repeat in range(2):
        cg,ca,rc=own(cpu.main);gg,ga,rg=own(lambda:gpu.main(args.device))
        widths=np.linspace(.1,10.,256)
        start=time.perf_counter();refs=[cpu.etch_levelset(w) for w in widths];tc=time.perf_counter()-start
        start=time.perf_counter();t,h,r=gpu.etch_batch(widths,device=args.device);tg=time.perf_counter()-start
        ca.update(batch_t=np.array([v[0] for v in refs]),batch_h=np.array([v[1] for v in refs]),batch_r=np.array([v[2] for v in refs]))
        ga.update(batch_t=np.broadcast_to(t,(256,len(t))).copy(),batch_h=h,batch_r=r)
        err=max(float(np.max(abs(ca[k]-ga[k])/(ATOL+RTOL*abs(ca[k])))) for k in ca)
        arrays={**{'cpu_'+k:v for k,v in ca.items()},**{'gpu_'+k:v for k,v in ga.items()}}
        np.savez_compressed(out/f'histories_{repeat}.npz',**arrays);captures.append(arrays)
        runs.append(dict(cpu_gates=cg,gpu_gates=gg,batch_size=256,cpu_s=tc,gpu_s=tg,parity=err,
            max_abs_error=max(float(np.max(abs(ca[k]-ga[k]))) for k in ca),
            all_pass=rc==rg==0 and all(cg.values()) and all(gg.values()) and err<=1))
    exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0])
    files=[Path(m.__file__) for m in (cpu,gpu,vof)]+[Path(__file__)]
    report=dict(device=str(wp.get_device(args.device)),atol=ATOL,rtol=RTOL,runs=runs,
        full_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
