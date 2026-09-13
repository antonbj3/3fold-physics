import numpy as np
import pytest
from physics_engine.quantum import quantum_revival as revival
from physics_engine.quantum import quantum_revival_gpu as rg
from physics_engine.quantum import bloch_oscillation as bloch
from physics_engine.quantum import bloch_oscillation_gpu as bg

@pytest.mark.parametrize('anharm',[0.,.02])
def test_revival_complete_time_batch_and_detuning(anharm):
    times=np.linspace(0.,1.2,130)
    expected=np.array([revival.autocorr(t,anharm=anharm) for t in times])
    np.testing.assert_allclose(rg.autocorr(times,anharm=anharm,device='cpu'),expected,atol=1e-12,rtol=1e-10)

@pytest.mark.parametrize('force',[.5,2.])
def test_wavepacket_unitarity_and_amplitude(force):
    value,state,xt=bg.bloch_amplitude(F=force,Nsite=30,w=5.,nt=71,device='cpu',return_history=True)
    np.testing.assert_allclose(value,bloch.bloch_amplitude(F=force,Nsite=30,w=5.,nt=71),atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(np.sum(abs(state)**2,axis=1),1.,atol=1e-12)
    np.testing.assert_allclose(xt,np.sum(np.arange(-30,31)[None,:]*abs(state)**2,axis=1),atol=1e-12)


def test_revival_series_changed_inputs_and_retained_outputs():
    series=rg.AutocorrelationSeries(device='cpu')
    saved=[]
    for E1,anharm,shape in [(3.7,0.,()),(3.7,0.,(7,)),(3.7,.02,(3,7)),(7.1,0.,(19,)),(3.7,0.,(0,)),(3.7,0.,(7,))]:
        times=np.linspace(-.2,1.3,int(np.prod(shape))).reshape(shape)
        actual=series(times,E1,anharm)
        expected=rg.autocorr(times,E1,anharm,device='cpu')
        np.testing.assert_array_equal(actual,expected)
        saved.append((actual,actual.copy()))
    for actual,expected in saved:np.testing.assert_array_equal(actual,expected)


def test_revival_series_copies_spectrum_inputs():
    series=rg.AutocorrelationSeries(device='cpu')
    times=np.linspace(0.,1.,31)
    expected=series(times).copy()
    saved=revival.CN2.copy()
    try:
        revival.CN2[:]=0.
        np.testing.assert_array_equal(series(times),expected)
        np.testing.assert_array_equal(rg.AutocorrelationSeries(device='cpu')(times),np.zeros(31))
    finally:revival.CN2[:]=saved


def test_revival_scalar_graph_changed_times_and_spectrum():
    graphed=rg.AutocorrelationSeries(device='cpu',graph_scalars=True)
    eager=rg.AutocorrelationSeries(device='cpu',graph_scalars=False)
    saved=[]
    for E1,anharm in [(3.7,0.),(3.7,.02),(7.1,.02),(3.7,0.)]:
        for t in [0.,-.3,.41,1.2]:
            actual=graphed(t,E1,anharm)
            expected=eager(t,E1,anharm)
            np.testing.assert_array_equal(actual,expected)
            saved.append((actual,actual.copy()))
        times=np.linspace(0.,1.,121)
        np.testing.assert_array_equal(graphed(times,E1,anharm),eager(times,E1,anharm))
    for actual,expected in saved:np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('capacity',[1,4])
def test_bloch_series_full_state_changed_force_width_grid_and_retained_outputs(capacity):
    series=bg.BlochSeries(device='cpu',max_operators=capacity)
    saved=[]
    for F,Nsite,w,nt in [(1.,30,5.,71),(1.,30,8.,51),(.5,30,5.,71),(2.,30,5.,71),(1.1,30,5.,71),(.9,30,5.,71),(1.,30,5.,71),(1.,20,5.,31),(1.,30,5.,71)]:
        actual=series(F,Nsite,w,nt,return_history=True)
        expected=bg.bloch_amplitude(F,Nsite,w,nt,device='cpu',return_history=True)
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
        saved.append((actual,tuple(np.copy(v) for v in actual)))
    for actual,expected in saved:
        for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)


def test_bloch_series_changed_hopping_and_spacing():
    series=bg.BlochSeries(device='cpu')
    original=(bloch.J,bloch.A)
    try:
        for hopping,spacing in [(1.,1.),(.8,1.),(.8,1.2),(1.,1.)]:
            bloch.J,bloch.A=hopping,spacing
            actual=series(Nsite=20,nt=31,return_history=True)
            expected=bg.bloch_amplitude(Nsite=20,nt=31,device='cpu',return_history=True)
            for a,b in zip(actual,expected):np.testing.assert_array_equal(a,b)
    finally:bloch.J,bloch.A=original
