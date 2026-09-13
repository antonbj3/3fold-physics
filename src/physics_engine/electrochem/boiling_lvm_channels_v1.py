"""Strict six-channel LVM reader and frozen heat-flux preservation observer."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import battery_boiling_curve_zuber_chf as frozen
ROOT=Path(__file__).resolve().parents[3]


class LVMContractError(ValueError):pass


def load_channels(path):
    """Read the single-heading, tab-separated layout of the pinned boiling files.

    Only technical metadata is retained. Operator names and absolute acquisition
    timestamps are neither returned nor written into public reports.
    """
    metadata={};samples=[];started=False;data_start=None
    with Path(path).open(encoding='latin-1',newline='') as stream:
        for line_number,fields in enumerate(csv.reader(stream,delimiter='\t'),1):
            if not fields or all(not f.strip() for f in fields):continue
            if not started:
                key=fields[0].strip()
                if key in ('Separator','Decimal_Separator','Multi_Headings','Channels','Y_Unit_Label'):
                    metadata[key]=[v.strip() for v in fields[1:] if v.strip()]
                if key=='X_Value':
                    expected=['X_Value']+['Temperature_'+str(i) for i in range(6)]
                    if fields[:7]!=expected or fields[7:] not in ([],['Comment']):raise LVMContractError('Unexpected channel labels')
                    started=True;data_start=line_number
                continue
            if len(fields)<7 or len(fields)>8:raise LVMContractError('Truncated or extra numeric channel')
            try:row=[float(x) for x in fields[:7]]
            except ValueError as error:raise LVMContractError('Non-numeric sample') from error
            samples.append(row)
    required={'Separator':['Tab'],'Decimal_Separator':['.'],'Multi_Headings':['No'],'Channels':['6'],'Y_Unit_Label':['Deg C']*6}
    if metadata!=required:raise LVMContractError('Unsupported or missing LVM metadata')
    values=np.asarray(samples,dtype=np.float64)
    if not started or values.ndim!=2 or values.shape[1]!=7 or len(values)<2:raise LVMContractError('Missing sample table')
    if not np.isfinite(values).all() or not (np.diff(values[:,0])>0).all():raise LVMContractError('Nonfinite sample or non-increasing time')
    return values,dict(columns=['X_Value']+['Temperature_'+str(i) for i in range(6)],temperature_unit='Deg C',data_start=data_start)


def digest(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def observation():
    paths=sorted((ROOT/'data/pool-boiling-multimodal').glob('*Temperature*.lvm'))
    if len(paths)!=4:raise LVMContractError('Four retained real temperature files required')
    records=[]
    for path in paths:
        full,metadata=load_channels(path)
        # NumPy's separate text reader checks every numeric column independently.
        oracle=np.loadtxt(path,delimiter='\t',skiprows=metadata['data_start'],usecols=range(7))
        legacy=frozen.load(path);old_q,old_dt=frozen.flux_superheat(legacy);q,dt=frozen.flux_superheat(full)
        gates=dict(full_array_oracle=digest(full)==digest(oracle),legacy_prefix=digest(full[:,:6])==digest(legacy),
                   frozen_flux=digest(q)==digest(old_q),frozen_superheat=digest(dt)==digest(old_dt),all_six_channels=full.shape[1]==7)
        records.append(dict(file=path.name,file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),shape=list(full.shape),
                            metadata=metadata,full_sha256=digest(full),oracle_sha256=digest(oracle),channel_sha256=[digest(full[:,i]) for i in range(7)],
                            legacy_sha256=digest(legacy),flux_sha256=digest(q),superheat_sha256=digest(dt),
                            retained_extra_channel=dict(name='Temperature_5',minimum=float(full[:,6].min()),maximum=float(full[:,6].max()),mean=float(full[:,6].mean())),gates=gates))
    gates=dict(four_files=len(records)==4,complete_independent_parse=all(r['gates']['full_array_oracle'] for r in records),
               frozen_parity=all(r['gates']['legacy_prefix'] and r['gates']['frozen_flux'] and r['gates']['frozen_superheat'] for r in records),
               complete_channels=all(r['gates']['all_six_channels'] for r in records))
    return dict(schema=1,gates=gates,records=records,status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
        sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(frozen.__file__))},
        unresolved=['Published Heat_flux.m unavailable locally; upstream dataset API returned HTTP403 during source audit',
                    'Physical depth assignment and surface-temperature reconstruction not independently verified against the published MATLAB file'],
        scope='Complete channel preservation and unchanged numerical mapping only; original boiling failures retained, no new heat-flux calibration or physical regime classification')


if __name__=='__main__':
    result=observation();(ROOT/'reports/boiling_lvm_channels_v1.json').write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result['gates']));raise SystemExit(int(not all(result['gates'].values())))
