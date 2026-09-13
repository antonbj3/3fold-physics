import numpy as np
import pytest
import torch
from physics_engine.process import marangoni_meltpool_cfd as cpu
from physics_engine.process.marangoni_meltpool_cfd_gpu import Pressure, Momentum

@pytest.mark.parametrize('shape',[(7,5),(12,8)])
def test_original_pinned_row_pressure_with_incompatible_rhs(shape):
    nx,nz=shape;dx=400e-6/nx;dz=220e-6/nz
    rhs=np.random.default_rng(17).normal(size=nx*nz)
    # Nonzero pin and incompatible sum catch a naive mean-subtracted solve.
    rhs[0]=3.
    expected=cpu.build_pressure(nx,nz,dx,dz).solve(rhs)
    actual=Pressure(nx,nz,dx,dz,'cpu').solve(rhs).numpy()
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-8)

def test_momentum_shear_sink_projection_first_steps():
    nx,nz=12,8;dx=400e-6/nx;dz=220e-6/nz
    x=np.arange(nx)[:,None];z=np.arange(nz)[None,:]
    T=300+1700*np.exp(-x/5-z/3)
    u=np.zeros((nx+1,nz));w=np.zeros((nx,nz+1));p=cpu.build_pressure(nx,nz,dx,dz)
    expected=cpu.relax_momentum(u,w,T,nx,nz,dx,dz,-.43e-3,p,max_sub=7)
    actual=Momentum('cpu')(u,w,T,nx,nz,dx,dz,-.43e-3,p,max_sub=7)
    for a,b in zip(actual[:2],expected[:2]):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)
    assert actual[2]==expected[2]


@pytest.mark.parametrize('shape',[(12,8),(31,17)])
@pytest.mark.parametrize('target',[35e-6,75e-6,120e-6])
def test_reused_conduction_calibration_preserves_bisection(shape,target):
    from physics_engine.process.marangoni_meltpool_cfd_gpu import tune_q0
    nx,nz=shape;dx=cpu.DOM_X/nx;dz=cpu.DOM_Z/nz
    expected=cpu.tune_q0(nx,nz,dx,dz,target)
    actual=tune_q0(nx,nz,dx,dz,target)
    assert actual==expected


def test_momentum_reuse_with_changed_temperature_and_tension():
    import os
    device=os.environ.get('PHYSICS_TEST_DEVICE','cpu')
    nx,nz=12,8;dx=cpu.DOM_X/nx;dz=cpu.DOM_Z/nz
    x=np.arange(nx)[:,None];z=np.arange(nz)[None,:]
    T=300+1700*np.exp(-x/5-z/3)
    momentum=Momentum(device)
    pressure=cpu.build_pressure(nx,nz,dx,dz)
    for shift,dsdt in [(0.,-.43e-3),(75.,-.43e-3),(25.,-.70e-3)]:
        u=np.zeros((nx+1,nz));w=np.zeros((nx,nz+1))
        expected=cpu.relax_momentum(u.copy(),w.copy(),T+shift,nx,nz,dx,dz,dsdt,pressure,max_sub=7)
        actual=momentum(u,w,T+shift,nx,nz,dx,dz,dsdt,pressure,max_sub=7)
        for a,b in zip(actual[:2],expected[:2]):np.testing.assert_allclose(a,b,atol=1e-10,rtol=1e-8)
        assert actual[2]==expected[2]
