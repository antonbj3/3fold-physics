"""Local stability is insufficient; retain global variation and signed nulls."""
from pathlib import Path
import sys
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/physics_engine/electrochem'))
from boiling_stationarity_v1 import analyse


def test_stable_windows_do_not_override_whole_record():
    r=analyse(np.arange(8),np.repeat([1.,2.],4),4)
    assert r['stable_block_count']==2 and not r['full_record_admitted']
    assert r['within_variance']==0 and r['between_variance']==r['total_variance']==.25
    assert r['decomposition_pass']


def test_stationary_positive_and_signed_null():
    assert analyse(np.arange(8),np.ones(8),4)['full_record_admitted']
    r=analyse(np.arange(8),-np.ones(8),4)
    assert r['mean']==-1 and not r['full_record_admitted'] and r['stable_block_count']==0


@pytest.mark.parametrize('mutation',('partial_block','time','nan','shape'))
def test_invalid_inputs_refused(mutation):
    t=np.arange(8,dtype=float);q=np.ones(8)
    if mutation=='partial_block':t=t[:-1];q=q[:-1]
    if mutation=='time':t[1]=t[0]
    if mutation=='nan':q[0]=np.nan
    if mutation=='shape':q=q.reshape(2,4)
    with pytest.raises(ValueError):analyse(t,q,4)
