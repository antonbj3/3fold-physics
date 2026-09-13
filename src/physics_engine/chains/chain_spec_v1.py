"""Closed-registry DAG execution, deterministic generation, propagation and nulls.

Specs contain modules, argument seams and declared controls, never Python code.
Numerical stage certificates do not transfer empirical scope to a new chain.
"""
import hashlib
import inspect
import json
from pathlib import Path
import re
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
import litho_chain_v1 as litho
import litho_tail_probe_v1 as pore_probe
import process_chain_v1 as process


def random_generator(seed):
    return np.random.default_rng(seed)


def uniforms(rng,like):
    return rng.random(len(like))


def radii(rng,like,median,log_sigma):
    return rng.lognormal(np.log(median),log_sigma,len(like))


def litho_summary(image,nsrc,resist,etched):
    contrast=float((image.max()-image.min())/(image.max()+image.min()+1e-30))
    if resist is None:return dict(line=False,contrast=contrast,nsrc=nsrc)
    L,W,sig_edge,ils=resist
    cd_e,ler_e,depth,ar=etched
    return dict(line=True,cd_resist=float(np.mean(L)),ler_resist=float(3.*np.std(L)/np.sqrt(2.)),
        cd_etch=cd_e,ler_etch=ler_e,sig_edge=sig_edge,ils=ils,depth=depth,ar=ar,contrast=contrast,nsrc=nsrc)


def width_summary(widths):
    if widths is None:return dict(line=False)
    return dict(line=True,cd=float(widths.mean()),ler=float(3.*widths.std()/np.sqrt(2.)),p99=pore_probe.tail(widths))


OPS={
 'litho_chain_v1.make_mask':litho.make_mask,
 'litho_chain_v1.aerial':litho.aerial,
 'litho_chain_v1.resist_lines':litho.resist_lines,
 'litho_chain_v1.etch_feature':litho.etch_feature,
 'process_chain_v1.thermal_field':process.thermal_field,
 'process_chain_v1.run_state':process.run_state,
 'process_chain_v1.evaluation':process.evaluation,
 'litho_tail_probe_v1.etched_lines':pore_probe.etched_lines,
 'litho_tail_probe_v1.pore':pore_probe.pore,
 'chain_spec_v1.random_generator':random_generator,
 'chain_spec_v1.uniforms':uniforms,
 'chain_spec_v1.radii':radii,
 'chain_spec_v1.litho_summary':litho_summary,
 'chain_spec_v1.width_summary':width_summary,
}
CACHEABLE={'process_chain_v1.thermal_field','process_chain_v1.run_state','litho_chain_v1.aerial'}


def plain(value):
    if isinstance(value,np.ndarray):
        return dict(dtype=str(value.dtype),shape=list(value.shape),sha256=hashlib.sha256(value.tobytes()).hexdigest())
    if isinstance(value,np.random.Generator):return plain(value.bit_generator.state)
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [plain(v) for v in value]
    return value


def canonical(value):
    return json.dumps(plain(value),sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def validate(spec):
    if spec.get('schema')!=1 or not re.fullmatch(r'[a-z][a-z0-9_]*',spec.get('name','')):
        raise ValueError('Invalid schema or chain name')
    parameters=spec['parameters'];seen=set()
    def binding(value):
        if isinstance(value,dict):
            if set(value)=={'param'}:
                if value['param'] not in parameters:raise ValueError('Unknown parameter')
            elif set(value)=={'ref'}:
                if value['ref'].split('.')[0] not in seen:raise ValueError('Forward, cyclic or missing seam')
            else:
                for v in value.values():binding(v)
        elif isinstance(value,list):
            for v in value:binding(v)
    for node in spec['nodes']:
        if not re.fullmatch(r'[a-z][a-z0-9_]*',node['id']) or node['id'] in seen:raise ValueError('Duplicate or invalid node')
        if node['op'] not in OPS:raise ValueError('Unknown stage operation')
        binding(node['kwargs']);binding(node.get('skip_if_none',[]))
        inspect.signature(OPS[node['op']]).bind(**node['kwargs'])
        seen.add(node['id'])
    if not seen:raise ValueError('Empty graph')
    binding(spec['output'])
    u=spec['uncertainty']
    if u['draws']<64 or not u['coordinates']:raise ValueError('At least64 uncertainty draws and coordinates required')
    for c in u['coordinates']:
        if not c['relative'] or not c['group']:raise ValueError('Empty uncertainty coordinate')
        if any(k not in parameters or not np.isfinite(v) or v<0 for k,v in c['relative'].items()):raise ValueError('Invalid uncertainty parameter')
    for null in spec['nulls']:
        if set(null['parameters'])-set(parameters):raise ValueError('Unknown null parameter')
        if null['comparison'] not in ('eq','lt_nominal','gt_nominal'):raise ValueError('Unknown null predicate')
    canonical(spec)
    return spec


def select(value,path):
    for key in path:
        if value is None:return None
        value=value[int(key)] if isinstance(value,(tuple,list)) else value[key]
    return value


def execute(spec,overrides=None,cache=None):
    validate(spec)
    params=dict(spec['parameters']);overrides=overrides or {}
    if set(overrides)-set(params):raise ValueError('Unknown override')
    params.update(overrides);values={};trace={}
    def resolve(value):
        if isinstance(value,dict):
            if set(value)=={'param'}:return params[value['param']]
            if set(value)=={'ref'}:
                keys=value['ref'].split('.');return select(values[keys[0]],keys[1:])
            return {k:resolve(v) for k,v in value.items()}
        if isinstance(value,list):return [resolve(v) for v in value]
        return value
    for node in spec['nodes']:
        kwargs=resolve(node['kwargs'])
        if any(resolve(v) is None for v in node.get('skip_if_none',[])):
            output=None
        else:
            key=(node['op'],sha(kwargs)) if cache is not None and node['op'] in CACHEABLE else None
            if key is not None and key in cache:output=cache[key]
            else:
                output=OPS[node['op']](**kwargs)
                if key is not None:cache[key]=output
        values[node['id']]=output;trace[node['id']]=sha(output)
    return dict(output=plain(resolve(spec['output'])),trace=trace)


def bundle(spec):
    validate(spec);cache={};nominal=execute(spec,cache=cache)
    u=spec['uncertainty'];coordinates=u['coordinates'];groups=sorted({c['group'] for c in coordinates})
    draws=np.random.default_rng(u['seed']).uniform(-1.,1.,(u['draws'],len(coordinates)))
    runs={};spreads={}
    for group in groups+['joint']:
        rows=[]
        for draw in draws:
            params={}
            for z,c in zip(draw,coordinates):
                if group in (c['group'],'joint'):
                    for name,fraction in c['relative'].items():params[name]=spec['parameters'][name]*(1.+float(z)*fraction)
            rows.append(execute(spec,params,cache))
        runs[group]=rows
        spreads[group]=[float(np.std([select(r['output'],key.split('.')) for r in rows])) for key in u['metrics']]
    joint=np.asarray(spreads['joint']);linear=np.sum([spreads[g] for g in groups],axis=0)
    dominant=[max(groups,key=lambda g:spreads[g][i]) for i in range(len(joint))]
    ratio=[float(a/b) if b else None for a,b in zip(joint,linear)]
    nulls={}
    for case in spec['nulls']:
        row=execute(spec,case['parameters'],cache);value=select(row['output'],case['metric'].split('.'))
        comparison=case['comparison'];base=select(nominal['output'],case['metric'].split('.'))
        accepted=(value==case['value']) if comparison=='eq' else (value<base if comparison=='lt_nominal' else value>base)
        nulls[case['name']]=dict(result=row,passed=bool(accepted))
    return dict(spec_sha256=sha(spec),scope=spec['scope'],nominal=nominal,draws=runs,
        composition=dict(metrics=u['metrics'],spreads=spreads,joint=joint.tolist(),linear_sum=linear.tolist(),ratio=ratio,
                         dominant=dominant,sublinear=(joint<=linear+1e-12).tolist()),nulls=nulls,
        gates=dict(nulls=all(x['passed'] for x in nulls.values()),finite_observations=all(np.isfinite(v).all() for v in spreads.values()),
            stage_contracts=all(all(row['output'].get('gates',{}).values()) for rows in runs.values() for row in rows)))


def generate(spec):
    validate(spec)
    embedded=json.dumps(spec,sort_keys=True,separators=(',',':'))
    return ('"""Generated from a chain specification; regenerate with chain_spec_v1."""\n'
        'import json\nfrom physics_engine.chains.chain_spec_v1 import write_bundle\n'
        'SPEC = json.loads('+repr(embedded)+')\n'
        "if __name__ == '__main__':\n    write_bundle(SPEC)\n").encode()


def write_bundle(spec):
    result=bundle(spec);path=Path('reports/chain_data')/(spec['name']+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(canonical(result)+b'\n')
    print(json.dumps(dict(name=spec['name'],gates=result['gates'],composition=result['composition'])))
    if not all(result['gates'].values()):raise SystemExit(1)


if __name__=='__main__':
    if len(sys.argv) not in (2,3):raise SystemExit('usage: chain_spec_v1.py SPEC.json [GENERATED.py]')
    spec=json.loads(Path(sys.argv[1]).read_text())
    if len(sys.argv)==3:Path(sys.argv[2]).write_bytes(generate(spec))
    else:write_bundle(spec)
