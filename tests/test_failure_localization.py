"""Delta-debugging controls distinguish interactions, invalid subsets and causes."""
from pathlib import Path
import sys
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src/physics_engine/chains'))
from failure_localization_v1 import minimize_failure,bisect_interval
from physics_failure_sites_v1 import gates


def test_single_and_interacting_witnesses():
    for required in ({'c'},{'b','d'}):
        def oracle(xs):return 'INVALID' if not xs else ('FAIL' if required<=set(xs) else 'PASS')
        result=minimize_failure(('a','b','c','d'),oracle)
        assert set(result['witness'])==required and result['deletion_minimal']
        assert result==minimize_failure(('a','b','c','d'),oracle)


def test_no_unique_site_and_invalid_removal():
    result=minimize_failure(('a','b'),lambda x:'FAIL' if x else 'INVALID')
    assert len(result['witness'])==1 and result['deletion_checks'][0]['verdict']=='INVALID'
    with pytest.raises(ValueError):minimize_failure(('a',),lambda x:'PASS')
    with pytest.raises(ValueError):minimize_failure(('a','a'),lambda x:'FAIL')


def test_interval_siblings_and_nonlocal_failure():
    result=bisect_interval(1024,lambda a,b:'FAIL' if a<=700<b else 'PASS',256)
    assert (result['start'],result['stop'])==(512,768)
    result=bisect_interval(1024,lambda a,b:'FAIL',256)
    assert all(r['verdict']=='FAIL' for r in result['trace'])
    result=bisect_interval(1024,lambda a,b:'FAIL' if b-a==1024 else 'PASS',256)
    assert result['stop']-result['start']==1024


def test_frozen_g3_predicate_targets_powered_noise():
    null=dict(order=0,tag='0',q_mean=-41.,q_std=500.,dt_mean=-7.)
    hot=dict(order=0,tag='CHF',q_mean=500.,q_std=30.,dt_mean=72.)
    assert gates([null,hot])['G3']=='FAIL'
    hot['q_std']=10.;assert gates([null,hot])['G3']=='PASS'
    null['q_mean']=-100000.;assert gates([null,hot])['G3']=='PASS'
