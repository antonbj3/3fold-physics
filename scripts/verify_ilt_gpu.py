"""Frozen ILT gates, complete 35-mask/focus image parity and same selected design."""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import time
import numpy as np
import torch
from physics_engine.litho import d_litho_certified_imaging_ilt as cpu
from physics_engine.litho import d_litho_certified_imaging_ilt_gpu as gpu
from physics_engine.litho import socs_batched_tcc_apply_gpu_api_for_ilt as socs

ROOT=Path(__file__).resolve().parents[1]
ATOL,RTOL=1e-9,1e-8


def compare(a,b):
    if isinstance(a,dict):
        return max([compare(a[k],b[k]) for k in a],default=0.) if isinstance(b,dict) and a.keys()==b.keys() else float('inf')
    if isinstance(a,list):
        return max([compare(x,y) for x,y in zip(a,b)],default=0.) if isinstance(b,list) and len(a)==len(b) else float('inf')
    if isinstance(a,bool) or a is None or isinstance(a,str): return 0. if a==b else float('inf')
    if not np.isfinite(a) or not np.isfinite(b): return float('inf')
    return float(abs(a-b)/(ATOL+RTOL*abs(a)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda',choices=['cuda','cpu'])
    p.add_argument('--out',default='reports/ilt_gpu_v1');args=p.parse_args()
    torch.set_num_threads(1)
    if args.device=='cuda' and not torch.cuda.is_available():raise SystemExit('CUDA required')
    out=ROOT/args.out;out.mkdir(parents=True,exist_ok=True)
    gpu.Imaging(args.device).batch(np.ones((1,256)),2.,.6)  # FFT/context warmup
    runs=[];captures=[];records=[]
    for repeat in range(2):
        with contextlib.redirect_stdout(io.StringIO()):
            anchors=[fn() for fn in (cpu.section_a_i,cpu.section_a_ii,cpu.section_a_iii,cpu.section_a_iv)]
            # Preserve the original kill gate before running ILT.
            if not all(r['PASS'] for r in anchors):
                (out/'anchor_failure.json').write_text(json.dumps(anchors,indent=2))
                return 2
            b=cpu.section_b();d=cpu.section_d();e=cpu.section_e()
            start=time.perf_counter();c=cpu.section_c();tc=time.perf_counter()-start
            if args.device=='cuda':torch.cuda.synchronize()
            start=time.perf_counter();g,images,masks,jacobian=gpu.section_c(args.device,return_jacobian=True)
            if args.device=='cuda':torch.cuda.synchronize()
            tg=time.perf_counter()-start
        src=cpu.source_line(256,2.,.6)
        reference=np.array([[cpu.aerial_1d(m,2.,.6,defocus=focus,source=src) for m in masks]
                            for focus in np.linspace(-90.,90.,9)])
        maximum=float(np.max(abs(images-reference)/(ATOL+RTOL*abs(reference))))
        records.append(dict(anchors=anchors,b=b,c_cpu=c,c_gpu=g,d=d,e=e))
        (out/f'full_record_{repeat}.json').write_text(json.dumps(records[-1],indent=2)+'\n')
        jacobian_reference=cpu.jacobian_1d(np.full(256,.5),2.,.6,src=src)
        jacobian_parity=float(np.max(abs(jacobian-jacobian_reference)/(ATOL+RTOL*abs(jacobian_reference))))
        captures.append(dict(cpu=reference,gpu=images,masks=masks,cpu_jacobian=jacobian_reference,gpu_jacobian=jacobian))
        np.savez_compressed(out/f'images_{repeat}.npz',**captures[-1])
        gates=dict(original_forward_anchors=all(r['PASS'] for r in anchors),
            original_b=b['PASS'],original_c_cpu=c['PASS'],original_c_gpu=g['PASS'],original_d=d['PASS'],
            full_candidate_images=maximum<=1,full_ilt_record=compare(c,g)<=1,
            full_jacobian=jacobian_parity<=1,
            selected_knobs_exact=c['ilt_knobs']==g['ilt_knobs'],
            process_windows_exact=all(c[k]==g[k] for k in ('pw_frac_naive','pw_frac_ilt')))
        runs.append(dict(cpu_s=tc,gpu_s=tg,image_parity=maximum,record_parity=compare(c,g),jacobian_parity=jacobian_parity,
            max_image_abs_error=float(np.max(abs(images-reference))),gates=gates))
    exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0]) and records[0]==records[1]
    files=[Path(cpu.__file__),Path(gpu.__file__),Path(socs.__file__),Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',
        atol=ATOL,rtol=RTOL,runs=runs,full_repeat_exact=exact,
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        all_pass=exact and all(all(r['gates'].values()) for r in runs))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__':raise SystemExit(main())
