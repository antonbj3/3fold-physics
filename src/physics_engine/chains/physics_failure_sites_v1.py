"""Locate two retained real-data gate failures without modifying their baselines."""
import hashlib
import itertools
import json
from pathlib import Path
import re
import subprocess
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src/physics_engine/wave_optics'))
sys.path.insert(0,str(ROOT/'src/physics_engine/electrochem'))
import l_kk_real_metamaterial as kk
import battery_boiling_curve_zuber_chf as boiling
from failure_localization_v1 import minimize_failure,bisect_interval


def digest(a):
    a=np.ascontiguousarray(a)
    return dict(shape=list(a.shape),dtype=str(a.dtype),sha256=hashlib.sha256(a.tobytes()).hexdigest())


def frozen_run(module):
    p=subprocess.run([sys.executable,module.__file__],cwd=ROOT,capture_output=True,timeout=120)
    lines=p.stdout.decode().splitlines()
    printed={m.group(1):('FAIL' if 'FAIL' in line else 'PASS') for line in lines if (m:=re.match(r'^\s*(G[1-4]) ',line))}
    return dict(returncode=p.returncode,printed_gates=printed,stdout_sha256=hashlib.sha256(p.stdout).hexdigest(),stderr_sha256=hashlib.sha256(p.stderr).hexdigest())


def kk_observation():
    files=sorted((ROOT/'data/phononic-metamaterial-transmission').glob('FIG4_Transmission_*.csv'))
    if len(files)!=5:raise RuntimeError('Five real KK fixtures required; no synthetic fallback')
    rows=[];arrays={}
    for name,df in kk.load_measurements():
        values=df.to_numpy(float);order=np.argsort(values[:,0]);w=values[order,0]
        chi=((values[:,1]-values[:,2])/2+1j*(values[:,3]-values[:,4])/2)[order]
        rows.append(dict(name=name,residual=kk.kk_residual(w,chi)['kk_residual'],frequency=digest(w),susceptibility=digest(chi),raw=digest(values),
                         finite=bool(np.isfinite(values).all()),strict_frequency=bool((np.diff(w)>0).all())))
        arrays[name]=(w,chi)
    w,chi=arrays[rows[0]['name']]
    scrambled=kk.kk_residual(w,chi.real+1j*np.random.default_rng(0).permutation(chi.imag))['kk_residual']
    by_name={r['name']:r for r in rows};threshold=.5*scrambled
    def oracle(names):
        if not names:return 'INVALID'
        return 'PASS' if np.median([by_name[n]['residual'] for n in names])<threshold else 'FAIL'
    names=list(by_name);singles=[dict(name=n,verdict=oracle([n])) for n in names]
    return dict(rows=rows,phase_scrambled_control=scrambled,threshold=threshold,median=float(np.median([r['residual'] for r in rows])),
        gate=oracle(names),witness=minimize_failure(names,oracle),singletons=singles,
        leave_one_out=[dict(removed=n,verdict=oracle([m for m in names if m!=n])) for n in names],
        site='gate_evaluation: median KK residual versus fixed original first-file scrambled control',
        diagnosis='No single measurement site' if all(r['verdict']=='FAIL' for r in singles) else 'Failing measurement subset localized',
        intervention='Only aggregate membership changes; full frequency grids, reconstruction and original scrambled control remain fixed',
        scope='The failing finite-band predicate does not establish acausality, a phase fault or a quadrature defect',frozen=frozen_run(kk))


def point(tag,a):
    q,dt=boiling.flux_superheat(a)
    row=dict(order=int(re.sub(r'\D','',tag) or 0),tag=tag,q_mean=float(q.mean()*10),q_std=float(q.std()*10),dt_mean=float(dt.mean()),
             samples=len(a),raw=digest(a),flux=digest(q),superheat=digest(dt),finite=bool(np.isfinite(a).all() and np.isfinite(q).all() and np.isfinite(dt).all()))
    return row,q,dt


def gates(points):
    points=sorted(points,key=lambda p:(p['order'],p['tag']))
    powered=[p for p in points if p['dt_mean']>0];null=[p for p in points if p['dt_mean']<=0]
    ordered=sorted(powered,key=lambda p:p['dt_mean'])
    g1='INVALID' if len(powered)<2 else ('PASS' if all(b['q_mean']>a['q_mean'] for a,b in zip(ordered,ordered[1:])) else 'FAIL')
    g2='INVALID' if not powered else ('PASS' if .5<max(p['q_mean'] for p in powered)/(boiling.zuber_chf()/1000)<1 else 'FAIL')
    g3='INVALID' if not powered or not null else ('PASS' if null[0]['dt_mean']<0 and powered[0]['q_std']<.05*powered[0]['q_mean'] else 'FAIL')
    return dict(G1=g1,G2=g2,G3=g3,G4='PASS' if g1==g2=='PASS' else 'FAIL')


def boiling_observation():
    files=sorted((ROOT/'data/pool-boiling-multimodal').glob('*Temperature*.lvm'))
    if len(files)!=4:raise RuntimeError('Four real boiling fixtures required; no synthetic fallback')
    points=[];arrays={}
    for path in files:
        tag=re.search(r'MC_(\w+)\.lvm',path.name).group(1);p,q,dt=point(tag,boiling.load(path));points.append(p);arrays[tag]=(q,dt)
    points.sort(key=lambda p:(p['order'],p['tag']));by_tag={p['tag']:p for p in points};names=list(by_tag);witnesses={}
    for gate in ('G1','G3'):
        oracle=lambda subset,g=gate:gates([by_tag[n] for n in subset])[g]
        witnesses[gate]=minimize_failure(names,oracle)
        witnesses[gate]['all_failing_pairs']=[list(pair) for pair in itertools.combinations(names,2) if oracle(pair)=='FAIL']
    powered=sorted((p for p in points if p['dt_mean']>0),key=lambda p:p['dt_mean'])
    inversions=[dict(lower_superheat=a['tag'],higher_superheat=b['tag'],flux_difference=b['q_mean']-a['q_mean'])
                for a,b in zip(powered,powered[1:]) if b['q_mean']<=a['q_mean']]
    selected=next(p for p in points if p['dt_mean']>0);q,_=arrays[selected['tag']]
    def interval(a,b):return 'PASS' if q[a:b].std()<.05*q[a:b].mean() else 'FAIL'
    time_site=bisect_interval(len(q),interval)
    for segment in [time_site,*time_site['trace']]:
        a,b=segment['start'],segment['stop'];segment['q_mean']=float(q[a:b].mean()*10);segment['q_std']=float(q[a:b].std()*10)
    synthetic=[point(tag,a)[0] for tag,a in boiling.synthetic_levels()]
    return dict(points=points,gates=gates(points),witnesses=witnesses,inversions=inversions,
        G3_selected_powered_tag=selected['tag'],G3_noise_ratio=selected['q_std']/selected['q_mean'],G3_time_site=time_site,
        site='G1 aggregate flux/superheat ordering; G3 standard-deviation predicate on first powered record',
        predicate_note='Frozen G3 checks powered-record noise and negative unpowered superheat, not near-zero unpowered heat flux',
        synthetic_control=gates(synthetic),frozen=frozen_run(boiling),
        scope='Data/predicate witnesses, not a diagnosed calibration, sensor mapping or physical boiling mechanism')


def observe():
    k=kk_observation();b=boiling_observation()
    sources=[Path(kk.__file__),Path(boiling.__file__),ROOT/'src/physics_engine/wave_optics/l_kk_susceptibility_cert.py',Path(__file__),Path(__file__).with_name('failure_localization_v1.py')]
    data=sorted((ROOT/'data/phononic-metamaterial-transmission').glob('*.csv'))+sorted((ROOT/'data/pool-boiling-multimodal').glob('*.lvm'))
    own=dict(original_failures=k['gate']=='FAIL' and b['gates']==dict(G1='FAIL',G2='PASS',G3='FAIL',G4='FAIL'),
             frozen_exit_parity=k['frozen']['returncode']==b['frozen']['returncode']==1 and b['gates']==b['frozen']['printed_gates'],
             input_integrity=all(r['finite'] and r['strict_frequency'] for r in k['rows']) and all(p['finite'] for p in b['points']),
             minimal_witnesses=k['witness']['deletion_minimal'] and all(w['deletion_minimal'] for w in b['witnesses'].values()),
             synthetic_boiling_control=all(v=='PASS' for v in b['synthetic_control'].values()))
    return dict(schema=1,kk=k,boiling=b,gates=own,sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        datasets={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in data},
        scope='Localization observer only; original real-data modules remain OWN-GATE-FAIL with unchanged thresholds')


if __name__=='__main__':
    result=observe();path=ROOT/'reports/physics_failure_sites_v1.json';path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps(result['gates']))
    raise SystemExit(int(not all(result['gates'].values())))
