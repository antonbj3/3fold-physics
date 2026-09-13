"""Refuse directional intensity spectra as inputs to a complex-response KK test."""
import csv
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import l_kk_susceptibility_cert as frozen
ROOT=Path(__file__).resolve().parents[3]
DIRECTIONAL_DOI='10.5061/dryad.dv41ns1wc'


class InputContractError(ValueError):pass


def checked_residual(frequency,response,metadata,*,evaluator=None):
    """Metadata admission is a declared contract, not independent calibration."""
    doi=str(metadata.get('dataset_doi','')).strip().lower()
    for prefix in ('https://doi.org/','http://doi.org/','doi:'):
        if doi.startswith(prefix):doi=doi[len(prefix):]
    if doi==DIRECTIONAL_DOI:
        raise InputContractError('Known directional-intensity dataset is not a complex-response measurement')
    if metadata.get('quantity')!='linear_complex_response' or metadata.get('complex_phase_observed') is not True:
        raise InputContractError('A declared complex linear response with observed phase is required')
    if metadata.get('configuration_axis')!='single_fixed_configuration':
        raise InputContractError('Response must refer to one fixed measurement configuration')
    unit=metadata.get('frequency_unit')
    if unit not in ('rad/s','Hz'):raise InputContractError('Explicit rad/s or Hz frequency unit required')
    w=np.asarray(frequency,dtype=float);chi=np.asarray(response)
    if w.ndim!=1 or chi.shape!=w.shape or len(w)<3 or not np.iscomplexobj(chi):
        raise InputContractError('Matching one-dimensional frequency and complex-response arrays required')
    if not np.isfinite(w).all() or not np.isfinite(chi).all() or not (w>0).all() or not (np.diff(w)>0).all():
        raise InputContractError('Finite positive strictly increasing frequency grid and finite response required')
    angular=w if unit=='rad/s' else 2*np.pi*w
    return (evaluator or frozen.kk_residual)(angular,chi)


def load_directional(directory,manifest):
    files=sorted(directory.glob('FIG4_Transmission_*.csv'))
    if {p.name for p in files}!=set(manifest['files']):raise InputContractError('Dataset inventory mismatch')
    if manifest['dataset_doi']!=DIRECTIONAL_DOI or manifest['semantics']['quantity']!='directional_intensity_transmission':
        raise InputContractError('Pinned dataset semantics mismatch')
    rows=[]
    for path in files:
        raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest['files'][path.name]:raise InputContractError('Dataset checksum mismatch')
        with path.open(newline='') as f:
            reader=csv.reader(f);header=next(reader);values=np.asarray([[float(x) for x in row] for row in reader],dtype=float)
        if values.ndim!=2 or values.shape[1]!=5 or not np.isfinite(values).all():raise InputContractError('Malformed directional table')
        rows.append((path.name,header,values))
    return rows


def observation():
    manifest_path=ROOT/'docs/KK_DATA_CONTRACT.json';manifest=json.loads(manifest_path.read_text())
    rows=load_directional(ROOT/'data/phononic-metamaterial-transmission',manifest)
    calls=[]
    def forbidden(*args):calls.append('invalid');raise AssertionError('Invalid dataset reached KK evaluator')
    refusals=[]
    metadata=dict(manifest['semantics'],dataset_doi=manifest['dataset_doi'])
    for name,header,values in rows:
        try:checked_residual(values[:,0],values[:,1:],metadata,evaluator=forbidden)
        except InputContractError as error:
            refusals.append(dict(file=name,reason=str(error),shape=list(values.shape),columns=header,
                raw_frequency_range=[float(values[0,0]),float(values[-1,0])],
                array_sha256=hashlib.sha256(values.tobytes()).hexdigest(),file_sha256=manifest['files'][name]))
        else:raise AssertionError('Directional measurement admitted')
    w=np.linspace(.001,40.,3000);chi=frozen.damped_oscillator_chi(w)
    valid=dict(quantity='linear_complex_response',complex_phase_observed=True,configuration_axis='single_fixed_configuration',frequency_unit='rad/s',source='synthetic_analytic_control')
    original=frozen.kk_residual(w,chi);accepted=checked_residual(w,chi,valid)
    relabeled=dict(valid,dataset_doi=DIRECTIONAL_DOI)
    try:checked_residual(w,chi,relabeled,evaluator=forbidden);relabel_refused=False
    except InputContractError:relabel_refused=True
    # Phase ambiguity control: identical intensities carry two distinct complex
    # responses. This is algebraic evidence, not a physical reconstruction.
    amplitude=np.sqrt(rows[0][2][:,1]);candidate_a=amplitude.astype(complex);candidate_b=1j*amplitude
    phase_ambiguous=bool(np.array_equal(np.abs(candidate_a)**2,np.abs(candidate_b)**2) and not np.array_equal(candidate_a,candidate_b))
    gates=dict(five_pinned_files=len(rows)==5,all_directional_refused=len(refusals)==5 and not calls,
               unchanged_positive=original==accepted and accepted['causal_pass'],known_dataset_relabel_refused=relabel_refused,
               intensity_phase_ambiguity=phase_ambiguous)
    return dict(schema=1,gates=gates,refusals=refusals,invalid_kk_calls=len(calls),synthetic_positive=accepted,
        positive_hashes=dict(frequency=hashlib.sha256(w.tobytes()).hexdigest(),response=hashlib.sha256(chi.tobytes()).hexdigest()),
        ambiguity_hashes=dict(first=hashlib.sha256(candidate_a.tobytes()).hexdigest(),second=hashlib.sha256(candidate_b.tobytes()).hexdigest()),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),Path(frozen.__file__),manifest_path)},
        status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL',
        scope='Input-refusal and synthetic-control certificate; no real-data KK causality conclusion. Legacy failed numerical report is retained but its measured-complex-response interpretation is invalid.')


if __name__=='__main__':
    report=observation();path=ROOT/'reports/kk_input_contract_v1.json'
    path.write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n');print(json.dumps(report['gates']))
    raise SystemExit(int(not all(report['gates'].values())))
