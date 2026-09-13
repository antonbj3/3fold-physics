"""Independent-process regeneration and frozen-oracle certificate for chain specs."""
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import tempfile
import numpy as np
import chain_spec_v1 as engine

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]


def direct(spec,overrides):
    p=dict(spec['parameters']);p.update(overrides)
    if spec['name']=='litho_from_spec':
        return engine.litho.run_chain(na=p['na'],sigma_out=p['sigma_out'],sigma_in=p['sigma_in'],dose=p['dose'],
            blur=p['blur'],etch_scale=p['etch_v0']/engine.litho.V0_ETCH,seed=p['seed'],clear_mask=p['clear_mask'],n_lines=p['n_lines'])
    field=engine.process.thermal_field(p['samples'])
    state=engine.process.run_state(field,power=p['power'],yield_scale=p['yield_scale'],amplitude=p['amplitude'],cooling=p['cooling'],relief=p['relief'])
    return engine.process.evaluation(state,p['strength_scale'])


def worker(name,destination):
    spec=json.loads((HERE/'specs'/(name+'.json')).read_text())
    generated=HERE/'generated'/(name+'.py')
    source_equal=generated.read_bytes()==engine.generate(spec)==engine.generate(json.loads(json.dumps(spec)))
    with tempfile.TemporaryDirectory(prefix='chain-spec-run-') as tmp:
        saved=Path.cwd()
        try:
            os.chdir(tmp)
            runpy.run_path(str(generated),run_name='__main__')
            result=json.loads((Path('reports/chain_data')/(name+'.json')).read_text())
        finally:os.chdir(saved)
    controls={}
    if name in ('litho_from_spec','process_from_spec'):
        cases=[('nominal',{})]+[(n['name'],n['parameters']) for n in spec['nulls']]
        # Each declared uncertainty coordinate also gets a direct-oracle control.
        for i,c in enumerate(spec['uncertainty']['coordinates']):
            cases.append(('coordinate_'+str(i),{k:spec['parameters'][k]*(1.+v) for k,v in c['relative'].items()}))
        for label,parameters in cases:
            actual=engine.execute(spec,parameters)['output'];expected=direct(spec,parameters)
            controls[label]=dict(exact=engine.canonical(actual)==engine.canonical(expected),
                actual_sha256=engine.sha(actual),reference_sha256=engine.sha(expected))
    else:
        without=engine.execute(spec,{'probability':0.})['output']
        original=json.loads((HERE/'specs/litho_from_spec.json').read_text())
        frozen=engine.execute(original)['output']
        controls['zero_pore_seam']=dict(exact=without['cd']==frozen['cd_etch'] and without['ler']==frozen['ler_etch'])
    result.update(source_equal=source_equal,oracle_controls=controls,
        generated_sha256=engine.hashlib.sha256(generated.read_bytes()).hexdigest())
    Path(destination).write_bytes(engine.canonical(result)+b'\n')


def main():
    if len(sys.argv)==4 and sys.argv[1]=='--worker':
        worker(sys.argv[2],sys.argv[3]);return 0
    if sys.argv[1:]:raise ValueError('Expected no arguments or --worker NAME OUTPUT')
    reports={};gates={}
    with tempfile.TemporaryDirectory(prefix='chain-spec-cert-') as tmp:
        for path in sorted((HERE/'specs').glob('*.json')):
            spec=json.loads(path.read_text());name=spec['name'];legs=[]
            for leg in range(2):
                target=Path(tmp)/(name+str(leg)+'.json')
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--worker',name,str(target)],check=True,
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1'),timeout=900)
                legs.append(target.read_bytes());print(name+' leg'+str(leg)+' complete',flush=True)
            first=json.loads(legs[0]);reports[name]=first
            gates[name+'_exact_repeat']=legs[0]==legs[1]
            gates[name+'_generated']=first['source_equal']
            gates[name+'_oracles']=all(c['exact'] for c in first['oracle_controls'].values())
            gates[name+'_contracts']=all(first['gates'].values())
    result=dict(schema=1,gates=gates,chains=reports,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')
    target=ROOT/'reports/chain_spec_certify_v1.json';target.parent.mkdir(exist_ok=True)
    target.write_bytes(engine.canonical(result)+b'\n')
    print(json.dumps(dict(gates=gates,status=result['status'])),flush=True)
    return 0 if all(gates.values()) else 1


if __name__=='__main__':raise SystemExit(main())
