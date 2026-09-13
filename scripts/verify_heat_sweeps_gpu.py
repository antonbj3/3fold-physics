"""Original NIST heat-source gates with measured inputs and complete field parity.

The two public input files are pinned in reports/process_gpu_inputs_v1 so the
existing Modal runner can use them without uploading unrelated datasets.
No synthetic fallback is admitted by this harness.
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
from physics_engine.process import weld_goldak_spot_nist as spot
from physics_engine.process import weld_goldak_spot_nist_gpu as sg
from physics_engine.process import lpbf_meltpool_render_match as lpbf
from physics_engine.process import lpbf_meltpool_render_match_gpu as lg
ROOT=Path(__file__).resolve().parents[1]
INPUT=ROOT/'reports/process_gpu_inputs_v1'
ATOL,RTOL=1e-8,1e-9


def suite(name,device,reference):
    module=spot if name=='spot' else lpbf
    if reference:
        scope=dict(vars(module))
        if name=='spot':
            scope['D']=str(INPUT)
            for key in ('load_tda','load_tdw'):
                fn=getattr(spot,key);scope[key]=types.FunctionType(fn.__code__,scope,key,fn.__defaults__)
    else:scope=sg.solver_scope(device,INPUT) if name=='spot' else lg.solver_scope(device)
    fn=types.FunctionType(module.main.__code__,scope,module.main.__name__)
    target=([spot.temp_field.__code__] if reference else [sg.temp_field.__code__]) if name=='spot' else (
        [lpbf.dT_eagar_tsai.__code__,lpbf.dT_rosenthal.__code__] if reference else [lg.dT_eagar_tsai.__code__,lg.dT_rosenthal.__code__])
    fields=[];gates={}
    def observer(frame,event,arg):
        if event=='return' and frame.f_code in target:fields.append(np.asarray(arg).copy())
        if event=='return' and frame.f_code==module.main.__code__:
            gates.update({key:bool(frame.f_locals[key]) for key in (('g1','g2','g3') if name=='spot' else ('g1','g2','g3','g4','g5'))})
    prev=sys.getprofile()
    if device=='cuda' and not reference:torch.cuda.synchronize()
    start=time.perf_counter()
    try:
        sys.setprofile(observer)
        with contextlib.redirect_stdout(io.StringIO()):rc=fn()
    finally:sys.setprofile(prev)
    if device=='cuda' and not reference:torch.cuda.synchronize()
    return fields,gates,rc,time.perf_counter()-start


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cpu','cuda'])
    p.add_argument('--out',default='reports/heat_sweeps_gpu_v1');args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    pins=json.loads((INPUT/'manifest.json').read_text())
    for key,digest in pins.items():
        if hashlib.sha256((INPUT/key).read_bytes()).hexdigest()!=digest:raise SystemExit('input hash mismatch')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    sg.temp_field(np.array([.001]),np.array([0.]),np.array([0.,.002]),np.array([300.,300.]),spot.AL,device=args.device)
    rows=[]
    for name in ('spot','lpbf'):
        captures=[];runs=[]
        for repeat in range(2):
            ca,cg,rc,tc=suite(name,args.device,True);ga,gg,rg,tg=suite(name,args.device,False)
            arrays={**{f'cpu_{i}':a for i,a in enumerate(ca)},**{f'gpu_{i}':a for i,a in enumerate(ga)}}
            captures.append(arrays);np.savez_compressed(out/f'{name}_{repeat}.npz',**arrays)
            same=len(ca)==len(ga) and all(a.shape==b.shape for a,b in zip(ca,ga))
            err=max(float(np.max(abs(a-b)/(ATOL+RTOL*abs(a)))) for a,b in zip(ca,ga)) if same else float('inf')
            runs.append(dict(cpu_gates=cg,gpu_gates=gg,cpu_s=tc,gpu_s=tg,field_count=len(ca),parity=err,
                max_abs_error=max(float(np.max(abs(a-b))) for a,b in zip(ca,ga)) if same else None,
                all_pass=rc==rg==0 and all(cg.values()) and all(gg.values()) and err<=1))
        exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0])
        rows.append(dict(module=name,runs=runs,full_repeat_exact=exact,all_pass=exact and all(r['all_pass'] for r in runs)))
    files=[Path(m.__file__) for m in (spot,sg,lpbf,lg)]+[Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',
        input_sha256=pins,atol=ATOL,rtol=RTOL,rows=rows,all_pass=all(r['all_pass'] for r in rows),
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
