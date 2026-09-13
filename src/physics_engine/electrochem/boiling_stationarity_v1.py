"""Full-record stationarity cannot be inferred from stable short windows."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import boiling_lvm_channels_v1 as reader
import battery_boiling_curve_zuber_chf as frozen
ROOT=Path(__file__).resolve().parents[3]
LIMIT=.05


def analyse(time,flux,block_samples=1024):
    time=np.asarray(time,dtype=np.float64);flux=np.asarray(flux,dtype=np.float64)
    if time.ndim!=1 or flux.shape!=time.shape or block_samples<2 or len(flux)==0 or len(flux)%block_samples:
        raise ValueError('Complete equal-sized blocks and matching one-dimensional arrays required')
    if not np.isfinite(time).all() or not np.isfinite(flux).all() or not (np.diff(time)>0).all():
        raise ValueError('Finite arrays and strictly increasing time required')
    blocks=flux.reshape(-1,block_samples);means=blocks.mean(1);variances=blocks.var(1)
    mean=float(flux.mean());total=float(flux.var());within=float(variances.mean());between=float(means.var())
    error=abs(total-within-between);budget=64*np.finfo(np.float64).eps*max(total,within+between,1.)
    full_pass=mean>0 and np.sqrt(total)<LIMIT*mean
    local=(means>0)&(np.sqrt(variances)<LIMIT*means)
    return dict(samples=len(flux),block_samples=block_samples,blocks=len(means),flux_unit='W/cm2',
        mean=mean,standard_deviation=float(np.sqrt(total)),coefficient_of_variation=float(np.sqrt(total)/abs(mean)) if mean else None,
        full_record_admitted=bool(full_pass),stable_block_count=int(local.sum()),
        within_variance=within,between_variance=between,total_variance=total,
        between_fraction=between/total if total else 0.,decomposition_error=error,decomposition_budget=budget,
        decomposition_pass=bool(error<=budget),
        block_arrays=dict(start_time=time[::block_samples].tolist(),end_time=time[block_samples-1::block_samples].tolist(),
                          mean=means.tolist(),standard_deviation=np.sqrt(variances).tolist(),admitted=local.tolist()),
        time_sha256=reader.digest(time),flux_sha256=reader.digest(flux),
        criterion='Full record requires positive mean and std < 0.05 * mean; locally admitted windows cannot override it')


def acquisition_block(path):
    with path.open(encoding='latin-1') as stream:
        for line in stream:
            fields=line.strip().split('\t')
            if fields[0]=='Samples':
                values=[int(x) for x in fields[1:] if x]
                if values!=[1024]*6:raise ValueError('Unexpected acquisition block metadata')
                return 1024
    raise ValueError('Missing acquisition block metadata')


def observation():
    prior=json.loads((ROOT/'reports/boiling_lvm_channels_v1.json').read_text());prior_by_file={r['file']:r for r in prior['records']}
    rows=[]
    for path in sorted((ROOT/'data/pool-boiling-multimodal').glob('*Temperature*.lvm')):
        a,_=reader.load_channels(path);q,dt=frozen.flux_superheat(a);row=analyse(a[:,0],q,acquisition_block(path))
        old=prior_by_file[path.name]
        row.update(file=path.name,tag=path.stem.split('_')[-1],superheat_mean=float(dt.mean()),
                   file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                   frozen_array_parity=reader.digest(q)==old['flux_sha256'] and reader.digest(dt)==old['superheat_sha256'])
        rows.append(row)
    by_tag={r['tag']:r for r in rows};eligible=sorted((r for r in rows if r['full_record_admitted'] and r['superheat_mean']>0),key=lambda r:r['superheat_mean'])
    conditional_order=len(eligible)>=2 and all(b['mean']>a['mean'] for a,b in zip(eligible,eligible[1:]))
    toy=analyse(np.arange(2048),np.repeat([1.,2.],1024))
    gates=dict(four_complete_records=len(rows)==4 and set(by_tag)=={'0','15','105','CHF'},
        frozen_full_array_parity=all(r['frozen_array_parity'] for r in rows),
        variance_decomposition=all(r['decomposition_pass'] for r in rows),
        stable_numeric_levels=by_tag['15']['full_record_admitted'] and by_tag['105']['full_record_admitted'] and conditional_order,
        whole_chf_refused=not by_tag['CHF']['full_record_admitted'],
        signed_null_retained=by_tag['0']['mean']<0 and not by_tag['0']['full_record_admitted'],
        aggregation_counterexample=toy['stable_block_count']==2 and not toy['full_record_admitted'])
    return dict(schema=1,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',gates=gates,records=rows,
        conditional_order=dict(tags=[r['tag'] for r in eligible],passes=conditional_order,
                               scope='Ordering only after predefined whole-record stationarity admission; original all-record G1 failure remains'),
        synthetic_counterexample=toy,physical_mapping_verified=False,
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(reader.__file__),Path(frozen.__file__))},
        scope='Temporal statistics under the frozen unverified physical mapping. Every sample/window retained; no selected-window CHF estimate, saturation calibration or physical boiling-regime claim. Original G1/G3 failures remain unchanged.')


if __name__=='__main__':
    result=observation();(ROOT/'reports/boiling_stationarity_v1.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['gates']));raise SystemExit(int(not all(result['gates'].values())))
