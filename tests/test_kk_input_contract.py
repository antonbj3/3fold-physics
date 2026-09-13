"""Invalid measurement semantics must be rejected before numerical evaluation."""
import copy
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import pytest
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src/physics_engine/wave_optics'))
import kk_input_contract_v1 as contract


def valid():return dict(quantity='linear_complex_response',complex_phase_observed=True,configuration_axis='single_fixed_configuration',frequency_unit='rad/s')


@pytest.mark.parametrize('mutation',('directional','missing_phase','changing_configuration','unknown_unit','real_only','duplicate_frequency','nonfinite','known_doi','known_doi_url'))
def test_bad_input_never_reaches_kk(mutation):
    metadata=valid();w=np.array([1.,2.,3.]);chi=np.array([1+1j,2+1j,3+1j]);calls=[]
    if mutation=='directional':metadata['quantity']='directional_intensity_transmission'
    if mutation=='missing_phase':metadata['complex_phase_observed']=False
    if mutation=='changing_configuration':metadata['configuration_axis']='propagation_direction'
    if mutation=='unknown_unit':metadata['frequency_unit']='unknown'
    if mutation=='real_only':chi=chi.real
    if mutation=='duplicate_frequency':w[1]=w[0]
    if mutation=='nonfinite':chi[1]=complex(np.nan,0)
    if mutation=='known_doi':metadata['dataset_doi']=contract.DIRECTIONAL_DOI
    if mutation=='known_doi_url':metadata['dataset_doi']='https://doi.org/'+contract.DIRECTIONAL_DOI.upper()
    with pytest.raises(contract.InputContractError):contract.checked_residual(w,chi,metadata,evaluator=lambda *x:calls.append(x))
    assert not calls


def test_explicit_frequency_units_and_unchanged_response():
    metadata=valid();metadata['frequency_unit']='Hz';w=np.array([1.,2.,3.]);chi=np.array([1+1j,2+1j,3+1j]);observed=[]
    def evaluator(angular,response):observed.append((angular.copy(),response.copy()));return {'ok':True}
    assert contract.checked_residual(w,chi,metadata,evaluator=evaluator)=={'ok':True}
    assert np.array_equal(observed[0][0],2*np.pi*w) and np.array_equal(observed[0][1],chi)


def test_pinned_dataset_rejects_missing_and_modified_files(tmp_path):
    manifest=json.loads((ROOT/'docs/KK_DATA_CONTRACT.json').read_text());data=ROOT/'data/phononic-metamaterial-transmission'
    for name in manifest['files']:shutil.copyfile(data/name,tmp_path/name)
    assert len(contract.load_directional(tmp_path,manifest))==5
    path=next(tmp_path.iterdir());original=path.read_bytes();path.write_bytes(original+b'\n')
    with pytest.raises(contract.InputContractError,match='checksum'):contract.load_directional(tmp_path,manifest)
    path.unlink()
    with pytest.raises(contract.InputContractError,match='inventory'):contract.load_directional(tmp_path,manifest)
