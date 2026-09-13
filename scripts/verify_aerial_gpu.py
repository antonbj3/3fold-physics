"""Run the original scalar/DOF main bodies and gates with full-image parity.

Return observers capture images actually used by each original call. They do not
replace binary mask generation, grid cutoffs, resolution search or any gate.
Timing runs omit the observer and include setup, transfers and synchronizations.
"""
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
from physics_engine.litho import litho_aerial_image_scalar as scalar
from physics_engine.litho import litho_depth_of_focus_cert as focus
from physics_engine.litho import litho_aerial_image_scalar_gpu as sg
from physics_engine.litho import litho_depth_of_focus_cert_gpu as fg

ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-10,1e-9


def capture(fn, main_code, image_code, cuda):
    images=[]; gates={}
    def observer(frame,event,arg):
        if frame.f_code not in (main_code,image_code):return None
        if event=='return':
            if frame.f_code==main_code:
                gates.update({k:bool(frame.f_locals[k]) for k in ('g1','g2','g3','g4')})
            else:
                value=arg.detach().cpu().numpy() if cuda else frame.f_locals['I']
                images.append(value.copy())
        return observer
    previous=sys.gettrace()
    try:
        sys.settrace(observer)
        with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    finally:sys.settrace(previous)
    return np.array(images),gates,rc


def timed(fn,device):
    if device=='cuda':torch.cuda.synchronize()
    start=time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    if device=='cuda':torch.cuda.synchronize()
    return time.perf_counter()-start,rc


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--out',default='reports/aerial_gpu_v1');args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    sg.aerial_image(.6,.33,13.5e-9,.5,device=args.device)
    rows=[]
    for name,cpu,gpu,image_fn in [('scalar',scalar,sg,scalar.aerial_contrast),('focus',focus,fg,focus.aerial_contrast_defocus)]:
        captures=[];runs=[]
        for repeat in range(2):
            ca,cg,cr=capture(cpu.main,cpu.main.__code__,image_fn.__code__,False)
            ga,gg,gr=capture(lambda:gpu.main(args.device),cpu.main.__code__,sg.aerial_image.__code__,True)
            same_shape=ca.shape==ga.shape
            err=float(np.max(abs(ga-ca)/(ATOL+RTOL*abs(ca)))) if same_shape else float('inf')
            tc,rc=timed(cpu.main,'cpu');tg,rg=timed(lambda:gpu.main(args.device),args.device)
            np.savez_compressed(out/f'{name}_{repeat}.npz',cpu=ca,gpu=ga)
            captures.append((ca,ga))
            runs.append(dict(cpu_gates=cg,gpu_gates=gg,cpu_s=tc,gpu_s=tg,image_count=len(ca),
                parity=err,max_abs_error=float(np.max(abs(ga-ca))) if same_shape else None,
                all_pass=all(cg.values()) and all(gg.values()) and len(cg)==len(gg)==4 and
                    cr==gr==rc==rg==0 and err<=1))
        exact=all(np.array_equal(x,y) for x,y in zip(captures[0],captures[1]))
        rows.append(dict(module=name,runs=runs,full_images_repeat_exact=exact,
                         all_pass=exact and all(r['all_pass'] for r in runs)))
    files=[Path(m.__file__) for m in (scalar,focus,sg,fg)]+[Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',
        atol=ATOL,rtol=RTOL,rows=rows,all_pass=all(r['all_pass'] for r in rows),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
