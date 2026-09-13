"""Fresh CPU/GPU parity, frozen five-gate forward suite and FD adjoint checks.

Parity fixed before GPU measurement: atol=1e-9, rtol=1e-8 for every coefficient,
angular field, cross section, FD field and sampled envelope/ray. The normalized
parity number is max(abs(gpu-cpu)/(atol+rtol*abs(cpu))); <=1 passes.
Two full runs are archived, with exact repeat checks excluding wall times.
"""
import argparse
import contextlib
import hashlib
import io
import json
from pathlib import Path
import runpy
import time
import numpy as np
import torch
from physics_engine.scattering import spine_scatter2d_multiple_helmholtz_forward_C_gpu as gpu
from physics_engine.scattering import pa_adjoint_helmholtz_envelope as adj
from physics_engine.scattering import pa_adjoint_helmholtz_envelope_gpu as adj_gpu

ROOT = Path(__file__).resolve().parents[1]
FORWARD = ROOT / 'src/physics_engine/scattering/spine_scatter2d_multiple_helmholtz_forward_C.py'
ATOL, RTOL = 1e-9, 1e-8


def array(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)


def parity(a, b):
    a, b = array(a), array(b)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        return float('inf')
    return float(np.max(np.abs(a - b) / (ATOL + RTOL * np.abs(a)), initial=0))


def timed(fn, device):
    if device == 'cuda': torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    if device == 'cuda': torch.cuda.synchronize()
    return value, time.perf_counter() - start


def forward_suite(solve, far, cross, batch=None, batch_far=None):
    ring = [[3 * np.cos(2*np.pi*i/6), 3*np.sin(2*np.pi*i/6)] for i in range(6)]
    arrays = {}
    def cs(key, k, pos, angle, m, prepared=None, prepared_field=None):
        ss, se, A, ns = cross(k, pos, 1., angle, m) if prepared is None else prepared
        values = array(ss), array(se), array(A)
        for name, value in zip(('scat', 'ext', 'coefficients'), values):
            arrays[key + '_' + name] = value
        field = prepared_field if prepared_field is not None else far(A, ns, torch.as_tensor(pos, dtype=torch.float64, device=A.device)
                    if isinstance(A, torch.Tensor) else np.asarray(pos), k,
                    np.linspace(0, 2*np.pi, 720, endpoint=False))
        arrays[key + '_field'] = array(field)
        return float(values[0]), float(values[1])
    ss, _ = cs('single', 1.3, [[0., 0.]], .3, 20)
    from scipy.special import jv, hankel1
    ns = np.arange(-20, 21)
    closed = 4/1.3 * np.sum(np.abs(-jv(ns, 1.3)/hankel1(ns, 1.3))**2)
    g1 = abs(ss-closed)/closed
    ss, se = cs('ring', 1.1, ring, .4, 12)
    g2 = abs(ss-se)/abs(se)
    A, ns, pos = solve(1.1, ring, 1., .4, 12)
    f = array(far(A, ns, pos, 1.1, np.array([1.9])))
    A, ns, pos = solve(1.1, ring, 1., np.pi+1.9, 12)
    r = array(far(A, ns, pos, 1.1, np.array([np.pi+.4])))
    arrays.update(reciprocity_forward=f, reciprocity_reverse=r)
    g3 = float((np.abs(f-r)/np.abs(f))[0])
    sm = [cs('order'+str(m), 1.1, ring, .4, m)[0] for m in (8, 10, 12)]
    g4 = abs(sm[-1]-sm[-2])/abs(sm[-1])
    frequencies = np.linspace(.3, 3., 40)
    orders = [max(8, min(12, int(k)+7)) for k in frequencies]
    if batch is None:
        sweep = [cs('frequency'+str(i), float(k), ring, .4, orders[i]) for i,k in enumerate(frequencies)]
    else:
        sweep = [None] * len(frequencies)
        for order in sorted(set(orders)):
            indices = np.flatnonzero(np.asarray(orders) == order)
            ks = frequencies[indices]
            ss, se, A, ns = batch(ks, ring, 1., .4, order)
            # Retain an independent complete angular evaluation for capture.
            fields = batch_far(A, ns, torch.as_tensor(ring, dtype=torch.float64, device=A.device), ks,
                               np.linspace(0, 2*np.pi, 720, endpoint=False))
            for j, index in enumerate(indices):
                sweep[index] = cs('frequency'+str(index), float(frequencies[index]), ring, .4, order,
                                  prepared=(ss[j], se[j], A[j], ns), prepared_field=fields[j])
    g5 = max(abs(ss-se)/max(abs(se), 1e-30) for ss, se in sweep)
    errors = [float(v) for v in (g1, g2, g3, g4, g5)]
    return arrays, errors


def adjoint_suite(device):
    fields, metrics = {}, []
    radii = np.array([.04,.06,.08,.10,.13,.16,.20,.24,.28,.32,.36,.40])
    for k in (0., 10., 20., 40., 70.):
        L = adj.build_L(201, .005, k)
        cpu, tc = timed(lambda: adj.solve_adj(L, 201, (100,100)), 'cpu')
        z, tg = timed(lambda: adj_gpu.solve_adj(kappa=k, device=device), device)
        z = array(z)
        env_c = adj.shell_envelope(cpu, 201, .005, (100,100), radii)
        env_g = adj.shell_envelope(z, 201, .005, (100,100), radii)
        rs, ray_c = adj.ray_signed(cpu, 201, .005, (100,100), 0., .45)
        _, ray_g = adj.ray_signed(z, 201, .005, (100,100), 0., .45)
        nc, zc = adj.sign_changes(rs, ray_c)
        ng, zg = adj.sign_changes(rs, ray_g)
        rhs = np.zeros(201**2); rhs[100*201+100] = 1.
        residual = float(np.max(np.abs(L.T @ z.ravel()-rhs)))
        fields.update({f'{k}_cpu':cpu, f'{k}_gpu':z, f'{k}_env_cpu':env_c,
                       f'{k}_env_gpu':env_g, f'{k}_ray_cpu':ray_c, f'{k}_ray_gpu':ray_g})
        metrics.append(dict(kappa=k, cpu_s=tc, gpu_s=tg, parity=parity(cpu,z),
            envelope_parity=parity(env_c,env_g), ray_parity=parity(ray_c,ray_g),
            zero_parity=parity(zc,zg), sign_changes_cpu=nc, sign_changes_gpu=ng,
            residual_inf=residual))
    return fields, metrics


def main():
    p = argparse.ArgumentParser(); p.add_argument('--device', default='cuda', choices=['cpu','cuda'])
    p.add_argument('--out', default='reports/scattering_gpu_v1')
    p.add_argument('--batch-forward', action='store_true', help='batch the original independent frequency cases by truncation order')
    args=p.parse_args()
    torch.set_num_threads(1)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA required; CPU fallback is not GPU evidence')
    out=ROOT/args.out; out.mkdir(parents=True, exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()): cpu=runpy.run_path(str(FORWARD))
    # Warm both workloads once. Timings include host special functions, assembly,
    # transfers and all observation calls; FD CPU timing excludes matrix assembly
    # (a conservative comparison); GPU FD includes basis construction.
    solve=lambda *a: gpu.solve_MS(*a, device=args.device)
    cross=lambda *a: gpu.cross_sections(*a, device=args.device)
    batch=(lambda *a: gpu.cross_sections_batch(*a, device=args.device)) if args.batch_forward else None
    cross(1.1, cpu['ring'], 1., .4, 12)
    adj_gpu.solve_adj(device=args.device)
    runs=[]; captures=[]
    for repeat in range(2):
        (ca, ce), tc=timed(lambda: forward_suite(cpu['solve_MS'],cpu['far_field'],cpu['cross_sections']), 'cpu')
        (ga, ge), tg=timed(lambda: forward_suite(solve,gpu.far_field,cross,batch=batch,batch_far=gpu.far_field_batch if args.batch_forward else None),args.device)
        af, am=adjoint_suite(args.device)
        capture={**{'cpu_'+k:v for k,v in ca.items()}, **{'gpu_'+k:v for k,v in ga.items()}, **{'adj_'+k:v for k,v in af.items()}}
        np.savez_compressed(out/f'fields_{repeat}.npz', **capture)
        captures.append(capture)
        fp=max(parity(ca[k],ga[k]) for k in ca)
        original_names=list(cpu['gates'])
        gates={name:bool(c < limit and g < limit) for name,c,g,limit in
               zip(original_names,ce,ge,[1e-8,1e-6,1e-6,1e-6,1e-5])}
        gates.update(forward_full_array_parity=fp<=1,
            adjoint_full_array_parity=all(max(m['parity'],m['envelope_parity'],m['ray_parity'],m['zero_parity'])<=1 for m in am),
            adjoint_sign_changes=all(m['sign_changes_cpu']==m['sign_changes_gpu'] for m in am),
            adjoint_original_operator_residual=all(m['residual_inf']<1e-8 for m in am))
        runs.append(dict(forward_cpu_s=tc,forward_gpu_s=tg,forward_cpu_errors=ce,
            forward_gpu_errors=ge,forward_parity=fp,adjoint=am,gates=gates))
    exact=all(np.array_equal(captures[0][k],captures[1][k]) for k in captures[0])
    files=[FORWARD,Path(adj.__file__),Path(gpu.__file__),Path(adj_gpu.__file__),Path(__file__)]
    report=dict(device=torch.cuda.get_device_name() if args.device=='cuda' else 'CPU control only',
        batch_forward=args.batch_forward,torch=torch.__version__,atol=ATOL,rtol=RTOL,runs=runs,exact_full_array_repeat=exact,
        source_sha256={str(f.resolve().relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        array_count=len(captures[0]),array_bytes=sum(v.nbytes for v in captures[0].values()),
        all_pass=exact and all(all(r['gates'].values()) for r in runs))
    (out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if report['all_pass'] else 2

if __name__=='__main__': raise SystemExit(main())
