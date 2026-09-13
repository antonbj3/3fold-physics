import os
import numpy as np
import pytest
from physics_engine.quantum import ising_2d as cpu
from physics_engine.quantum import ising_2d_gpu as gpu

@pytest.mark.parametrize('batch',[1,4,32])
@pytest.mark.parametrize('L',[5,8])
@pytest.mark.parametrize('T',[1.,2.3,6.])
def test_exact_sublattice_updates_and_rng_consumption(L,T,batch):
    r0=np.random.default_rng(17);r1=np.random.default_rng(17)
    expected=cpu.simulate(L,T,r0,eq=7,meas=11)
    actual,states,M,E=gpu.simulate(L,T,r1,eq=7,meas=11,device=os.environ.get('PHYSICS_TEST_DEVICE','cpu'),return_history=True,random_batch_sweeps=batch)
    np.testing.assert_array_equal(actual,expected)
    assert r0.bit_generator.state==r1.bit_generator.state
    np.testing.assert_array_equal(M,np.abs(states.mean(axis=(1,2))))
    np.testing.assert_array_equal(E,-(states*(np.roll(states,1,axis=1)+np.roll(states,1,axis=2))).mean(axis=(1,2)))


@pytest.mark.parametrize('batch',[0,-1,1.5,True])
def test_invalid_batch_does_not_consume_rng(batch):
    rng=np.random.default_rng(19)
    state=rng.bit_generator.state
    with pytest.raises(ValueError,match='random_batch_sweeps'):
        gpu.simulate(4,2.,rng,device='cpu',random_batch_sweeps=batch)
    assert rng.bit_generator.state==state


def test_random_draw_storage_is_bounded():
    class ObservedRng:
        def __init__(self):
            self.rng=np.random.default_rng(23)
            self.shapes=[]
        def choice(self,*args,**kwargs):
            return self.rng.choice(*args,**kwargs)
        def random(self,shape):
            self.shapes.append(shape)
            return self.rng.random(shape)
    rng=ObservedRng()
    expected_rng=np.random.default_rng(23)
    expected=cpu.simulate(5,2.3,expected_rng,eq=7,meas=11)
    actual=gpu.simulate(5,2.3,rng,eq=7,meas=11,device='cpu',random_batch_sweeps=4)
    np.testing.assert_array_equal(actual,expected)
    assert rng.shapes==[(8,5,5)]*4+[(4,5,5)]
    assert rng.rng.bit_generator.state==expected_rng.bit_generator.state


@pytest.mark.parametrize('L',[5,8])
def test_scalar_only_matches_full_history_and_rng(L,monkeypatch):
    import os
    device=os.environ.get('PHYSICS_TEST_DEVICE','cpu')
    full_rng=np.random.default_rng(29)
    scalar_rng=np.random.default_rng(29)
    full=gpu.simulate(L,2.3,full_rng,eq=7,meas=11,device=device,
                      return_history=True,random_batch_sweeps=4)
    allocations=[]
    original_empty=gpu.wp.empty
    def observed_empty(shape,*args,**kwargs):
        allocations.append(shape)
        return original_empty(shape,*args,**kwargs)
    monkeypatch.setattr(gpu.wp,'empty',observed_empty)
    scalar=gpu.simulate(L,2.3,scalar_rng,eq=7,meas=11,device=device,
                        random_batch_sweeps=4)
    np.testing.assert_array_equal(scalar,full[0])
    assert scalar_rng.bit_generator.state==full_rng.bit_generator.state
    assert (11,L,L) not in allocations
    assert (1,1,1) in allocations


@pytest.mark.parametrize('eq',[0,7,8])
def test_block_boundaries_preserve_complete_odd_lattice_history(eq):
    import types
    expected_rng=np.random.default_rng(41);actual_rng=np.random.default_rng(41)
    states=[];calls=0
    def sweep(*args):
        nonlocal calls
        value=cpu.sweep(*args);calls+=1
        if calls>2*eq and calls%2==0:states.append(value.copy())
        return value
    scope=dict(vars(cpu));scope['sweep']=sweep
    reference=types.FunctionType(cpu.simulate.__code__,scope,cpu.simulate.__name__,cpu.simulate.__defaults__)
    expected=reference(5,2.3,expected_rng,eq=eq,meas=7)
    actual,history,_,_=gpu.simulate(5,2.3,actual_rng,eq=eq,meas=7,
        device=os.environ.get('PHYSICS_TEST_DEVICE','cpu'),return_history=True,random_batch_sweeps=4)
    np.testing.assert_array_equal(history,np.array(states))
    np.testing.assert_array_equal(actual,expected)
    assert actual_rng.bit_generator.state==expected_rng.bit_generator.state


@pytest.mark.parametrize('history',[False,True])
def test_series_resets_and_rebuilds_without_changing_previous_outputs(history):
    import copy
    series=gpu.SimulationSeries(device=os.environ.get('PHYSICS_TEST_DEVICE','cpu'),return_history=history,random_batch_sweeps=4)
    retained=[]
    for L,T,eq,meas,seed in [(5,1.2,7,9,3),(5,3.1,7,9,17),(6,2.3,0,5,11),(5,1.2,7,9,3)]:
        a=np.random.default_rng(seed);b=np.random.default_rng(seed)
        actual=series(L,T,a,eq,meas)
        expected=gpu.simulate(L,T,b,eq,meas,device=series.device,return_history=history,random_batch_sweeps=4)
        if history:
            assert actual[0]==expected[0]
            for x,y in zip(actual[1:],expected[1:]):np.testing.assert_array_equal(x,y)
        else:assert actual==expected
        assert a.bit_generator.state==b.bit_generator.state
        retained.append((actual,copy.deepcopy(actual)))
    for actual,saved in retained:
        if history:
            assert actual[0]==saved[0]
            for x,y in zip(actual[1:],saved[1:]):np.testing.assert_array_equal(x,y)
        else:assert actual==saved
