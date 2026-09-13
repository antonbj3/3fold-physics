"""Original revival/Bloch gates and full autocorrelation/complex propagation parity."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from physics_engine.quantum import quantum_revival as revival
from physics_engine.quantum import quantum_revival_gpu as rg
from physics_engine.quantum import bloch_oscillation as bloch
from physics_engine.quantum import bloch_oscillation_gpu as bg
ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-10,1e-8


def suite(name,device,reference):
    module=revival if name=='revival' else bloch
    port=rg if name=='revival' else bg
    arrays=[];gates={}
    def observer(frame,event,arg):
        if event!='return':return
        if frame.f_code==module.main.__code__:
            gates.update(g4=bool(frame.f_locals['g4']),g5=bool(frame.f_locals['g5']),scaffold=bool(frame.f_locals['res'].ok))
        if name=='revival' and frame.f_code==(revival.autocorr.__code__ if reference else rg.autocorr.__code__):
            arrays.append(np.atleast_1d(arg).copy())
        if name=='bloch' and frame.f_code==(bloch.bloch_amplitude.__code__ if reference else bg.bloch_amplitude.__code__):
            local=frame.f_locals
            if reference:
                state=np.array([local['V']@(np.exp(-1j*local['E']*t)*local['c']) for t in local['ts']])
                xt=local['xt']
            else:state=local['state'].detach().cpu().numpy();xt=local['xt'].detach().cpu().numpy()
            arrays.extend([state.copy(),xt.copy()])
    prev=sys.getprofile()
    if not reference and device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    try:
        sys.setprofile(observer)
        with contextlib.redirect_stdout(io.StringIO()):rc=module.main() if reference else port.main(device)
    finally:sys.setprofile(prev)
    if not reference and device=='cuda':torch.cuda.synchronize()
    seconds=time.perf_counter()-start
    return ([np.concatenate(arrays)] if name=='revival' else arrays),gates,rc,seconds


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/wavepacket_gpu_v1');args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    rg.autocorr(np.array([0.,1.]),device=args.device);bg.bloch_amplitude(Nsite=4,nt=3,device=args.device)
    rows=[]
    for name in ('revival','bloch'):
        runs=[];captures=[]
        for repeat in range(2):
            ca,cg,rc,tc=suite(name,args.device,True);ga,gg,rr,tg=suite(name,args.device,False)
            arrays={**{f'cpu_{i}':v for i,v in enumerate(ca)},**{f'gpu_{i}':v for i,v in enumerate(ga)}}
            captures.append(arrays);np.savez_compressed(out/f'{name}_{repeat}.npz',**arrays)
            same=len(ca)==len(ga) and all(a.shape==b.shape for a,b in zip(ca,ga))
            err=max(float(np.max(abs(a-b)/(ATOL+RTOL*abs(a)))) for a,b in zip(ca,ga)) if same else float('inf')
            runs.append(dict(cpu_gates=cg,gpu_gates=gg,cpu_s=tc,gpu_s=tg,parity=err,
                value_count=sum(v.size for v in ca),max_abs_error=max(float(np.max(abs(a-b))) for a,b in zip(ca,ga)) if same else None,
                all_pass=rc==rr==0 and all(cg.values()) and all(gg.values()) and err<=1))
        exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0])
        rows.append(dict(module=name,runs=runs,full_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs)))
    files=[Path(m.__file__) for m in (revival,rg,bloch,bg)]+[Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',atol=ATOL,rtol=RTOL,
        rows=rows,all_pass=all(r['all_pass'] for r in rows),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
