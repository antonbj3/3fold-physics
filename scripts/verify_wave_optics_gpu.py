"""Full optical propagation arrays, original gates and two exact hash repeats.

Every 4096x4096 complex Airy field is compared in memory; complete hashes and
explicit center-row slices are archived to avoid duplicating gigabytes. Other
propagated arrays are archived in full. The G3 executable threshold is preserved
as rel_err < 0.5, despite the CPU docstring's stricter 0.5% wording.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from physics_engine.wave_optics import wave_optics_cell as cpu
from physics_engine.wave_optics import wave_optics_cell_gpu as gpu
ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-12,1e-9


def suite(device,reference):
    arrays=[]
    module=cpu if reference else gpu
    def observer(frame,event,arg):
        if event!='return':return
        if frame.f_code==module.fraunhofer_1d.__code__:arrays.append(np.array(arg[1],copy=True))
        elif frame.f_code==module.gate_G3_circular_aperture.__code__:
            arrays.append(frame.f_locals['field'].copy())
            arrays.append(frame.f_locals['row'].copy())
        elif frame.f_code==module._coherent_image_intensity.__code__:arrays.append(np.array(arg[0],copy=True))
    prev=sys.getprofile()
    if not reference and device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    try:
        sys.setprofile(observer);result=cpu.run_all() if reference else gpu.run_all(device)
    finally:sys.setprofile(prev)
    if not reference and device=='cuda':torch.cuda.synchronize()
    return arrays,result,time.perf_counter()-start


def compare_records(a,b):
    if isinstance(a,dict):return max(compare_records(a[k],b[k]) for k in a) if a.keys()==b.keys() else float('inf')
    if isinstance(a,(list,tuple)):return max(compare_records(x,y) for x,y in zip(a,b)) if len(a)==len(b) else float('inf')
    if isinstance(a,(bool,np.bool_,str)):return 0. if a==b else float('inf')
    return float(abs(a-b)/(ATOL+RTOL*abs(a))) if np.isfinite(a) and np.isfinite(b) else float('inf')


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/wave_optics_gpu_v1');args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    gpu._fft2_shift(np.ones((32,32)),args.device)
    runs=[];manifests=[]
    for repeat in range(2):
        ca,cr,tc=suite(args.device,True);ga,gr,tg=suite(args.device,False)
        manifest={};archive={};errors=[]
        assert len(ca)==len(ga)
        for index,(a,b) in enumerate(zip(ca,ga)):
            if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():raise RuntimeError('invalid propagated field')
            err=0.;maximum=0.
            af,bf=a.reshape(-1),b.reshape(-1)
            for start in range(0,a.size,1<<20):
                x,y=af[start:start+(1<<20)],bf[start:start+(1<<20)]
                err=max(err,float(np.max(abs(x-y)/(ATOL+RTOL*abs(x)))))
                maximum=max(maximum,float(np.max(abs(x-y))))
            errors.append(err)
            for label,value in (('cpu',a),('gpu',b)):
                key=f'{label}_{index}'
                manifest[key]=dict(shape=list(value.shape),dtype=str(value.dtype),bytes=value.nbytes,
                    sha256=hashlib.sha256(value.tobytes()).hexdigest())
                archive[key]=value[value.shape[0]//2].copy() if value.ndim==2 else value.copy()
                manifest[key]['archive_scope']='center_row_only' if value.ndim==2 else 'complete'
            manifest[f'comparison_{index}']=dict(normalized_parity=err,max_abs_error=maximum)
        np.savez_compressed(out/f'archive_{repeat}.npz',**archive)
        (out/f'fields_{repeat}.json').write_text(json.dumps(manifest,indent=2)+'\n');manifests.append(manifest)
        (out/f'gate_records_{repeat}.json').write_text(json.dumps(dict(cpu=cr,gpu=gr),indent=2,default=lambda x:bool(x) if isinstance(x,np.bool_) else float(x))+'\n')
        gates=all(g['pass_'] for g in cr[:3]) and all(g['pass_'] for g in gr[:3])
        runs.append(dict(cpu_s=tc,gpu_s=tg,full_array_parity=max(errors),record_parity=compare_records(cr,gr),
            full_array_count=len(ca),full_bytes_per_leg=sum(a.nbytes for a in ca),
            all_pass=bool(gates and max(errors)<=1 and compare_records(cr,gr)<=1)))
        del ca,ga,archive,a,b,af,bf
    exact=manifests[0]==manifests[1]
    files=[Path(cpu.__file__),Path(gpu.__file__),Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',atol=ATOL,rtol=RTOL,
        runs=runs,complete_hashes_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
