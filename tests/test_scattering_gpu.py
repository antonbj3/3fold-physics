"""CPU backend controls catch transpose boundary errors and invalid geometries."""
import numpy as np
import pytest
import torch
from physics_engine.scattering.pa_adjoint_helmholtz_envelope import build_L
from physics_engine.scattering.pa_adjoint_helmholtz_envelope_gpu import HelmholtzOperator
from physics_engine.scattering.spine_scatter2d_multiple_helmholtz_forward_C_gpu import solve_MS

@pytest.mark.parametrize('kappa', [0., 10., 20., 40., 70.])
def test_full_operator_and_adjoint_with_boundary_rhs(kappa):
    n=17; h=1/(n-1)
    L=build_L(n,h,kappa).toarray()
    rng=np.random.default_rng(20260913)
    rhs=rng.normal(size=(n,n))
    op=HelmholtzOperator(n,h,kappa,device='cpu')
    for transpose in (False,True):
        actual=op.solve(rhs,adjoint=transpose).numpy()
        expected=np.linalg.solve(L.T if transpose else L,rhs.ravel()).reshape(n,n)
        np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-8)
    x=rng.normal(size=(n,n)); y=rng.normal(size=(n,n))
    left=np.vdot(op.solve(x).numpy(),y)
    right=np.vdot(x,op.solve(y,adjoint=True).numpy())
    np.testing.assert_allclose(left,right,atol=1e-9,rtol=1e-8)

def test_refuse_singular_interior_and_bad_source():
    n=17; h=1/(n-1)
    k=(8/h**2*np.sin(np.pi/(2*(n-1)))**2)**.5
    with pytest.raises(ValueError,match='singular'): HelmholtzOperator(n,h,k,device='cpu')
    with pytest.raises(ValueError): solve_MS(1., [[0.,0.],[1.,0.]],1.,0.,8,device='cpu')
    with pytest.raises(ValueError): solve_MS(float('nan'),[[0.,0.]],1.,0.,8,device='cpu')


@pytest.fixture(scope='module')
def original_forward():
    import contextlib,io,runpy
    from pathlib import Path
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path(str(Path(__file__).resolve().parents[1]/'src/physics_engine/scattering/spine_scatter2d_multiple_helmholtz_forward_C.py'))

@pytest.mark.parametrize('order',[0,1,4,12])
def test_asymmetric_cylinders_and_order_extremes(original_forward,order):
    import os
    from physics_engine.scattering import spine_scatter2d_multiple_helmholtz_forward_C_gpu as gpu
    positions=np.array([[0.,0.],[3.2,.4],[-.7,3.8]])
    expected,ns,pos=original_forward['solve_MS'](1.3,positions,.7,.37,order)
    actual,modes,points=gpu.solve_MS(1.3,positions,.7,.37,order,device=os.environ.get('PHYSICS_TEST_DEVICE','cpu'))
    np.testing.assert_allclose(actual.cpu().numpy(),expected,atol=1e-9,rtol=1e-8)
    np.testing.assert_array_equal(modes.cpu().numpy(),ns)
    angles=np.array([-.2,.37,1.9,3.7])
    np.testing.assert_allclose(gpu.far_field(actual,modes,points,1.3,angles).cpu().numpy(),original_forward['far_field'](expected,ns,pos,1.3,angles),atol=1e-9,rtol=1e-8)


@pytest.mark.parametrize('count',[1,3])
@pytest.mark.parametrize('order',[0,4])
def test_frequency_batch_keeps_independent_cases(count,order):
    import os
    from physics_engine.scattering import spine_scatter2d_multiple_helmholtz_forward_C_gpu as gpu
    device=os.environ.get('PHYSICS_TEST_DEVICE','cpu')
    ks=np.array([.3,1.1,2.8])[:count];positions=np.array([[0.,0.],[3.2,.4],[-.7,3.8]])
    actual=gpu.cross_sections_batch(ks,positions,.7,.37,order,device=device)
    expected=[gpu.cross_sections(float(k),positions,.7,.37,order,device=device) for k in ks]
    for index in range(3):
        np.testing.assert_allclose(actual[index].cpu().numpy(),np.stack([r[index].cpu().numpy() for r in expected]),atol=1e-9,rtol=1e-8)
    for r in expected:np.testing.assert_array_equal(actual[3].cpu().numpy(),r[3].cpu().numpy())
    points=torch.as_tensor(positions,dtype=torch.float64,device=device);angles=np.array([-.2,.37,1.9,3.7])
    fields=gpu.far_field_batch(actual[2],actual[3],points,ks,angles).cpu().numpy()
    for i,k in enumerate(ks):
        expected_field=gpu.far_field(expected[i][2],expected[i][3],points,float(k),angles).cpu().numpy()
        np.testing.assert_allclose(fields[i],expected_field,atol=1e-9,rtol=1e-8)

@pytest.mark.parametrize('ks',[[],[-1.],[float('nan')],[[1.]]])
def test_batch_refuses_invalid_frequency_vectors(ks):
    from physics_engine.scattering import spine_scatter2d_multiple_helmholtz_forward_C_gpu as gpu
    with pytest.raises(ValueError):gpu.solve_MS_batch(ks,[[0.,0.]],1.,0.,4,device='cpu')
