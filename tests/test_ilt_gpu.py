import numpy as np
import pytest
from physics_engine.litho import d_litho_certified_imaging_ilt as cpu
from physics_engine.litho.d_litho_certified_imaging_ilt_gpu import Imaging, kernels

@pytest.mark.parametrize('focus', [-90.,0.,90.])
def test_complex_focus_and_nonuniform_source(focus):
    src=([-2,0,1],np.array([1.,3.,2.]))
    n=64; dx=8.; phi,lam=kernels(n,dx,.6,focus,src)
    P=cpu.pupil_1d(n,dx,defocus=focus)
    A=np.array([np.roll(P,-b) for b in src[0]])
    expected=A.T@(src[1][:,None]*A.conj())/src[1].sum()
    np.testing.assert_allclose(phi.T@(lam[:,None]*phi.conj()),expected,atol=1e-13)
    masks=np.random.default_rng(17).random((3,n))
    actual=Imaging('cpu').batch(masks,dx,.6,focus,src)
    expected=np.array([cpu.aerial_1d(m,dx,.6,defocus=focus,source=src) for m in masks])
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)

def test_invalid_sources_refused():
    with pytest.raises(ValueError): kernels(64,8.,.6,source=([],[]))
    with pytest.raises(ValueError): kernels(64,8.,.6,source=([0],[-1.]))


@pytest.mark.parametrize('dtype',[np.float32,np.float64,np.int64])
@pytest.mark.parametrize('values,threshold',[
    ([],0.),([1.],1.),([0.,0.,1.,0.,-1.,0.],0.),
    ([-2.,1.,-1.,2.,-2.],0.3),([1.,1.,1.],1.),
    ([0.,1.,2.,3.,2.,1.,0.],2.)])
def test_vectorized_edges_preserve_scalar_crossings(dtype,values,threshold):
    from physics_engine.litho.d_litho_certified_imaging_ilt_gpu import find_edges_1d
    image=np.array(values,dtype=dtype)
    expected=cpu.find_edges_1d(image,2.3,threshold)
    actual=find_edges_1d(image,2.3,threshold)
    np.testing.assert_array_equal(actual,expected)


def test_vectorized_edges_preserve_nan_and_infinite_neighbors():
    from physics_engine.litho.d_litho_certified_imaging_ilt_gpu import find_edges_1d
    image=np.array([0.,np.nan,-np.inf,np.inf,0.,1.,-1.])
    with np.errstate(invalid='ignore'):
        expected=cpu.find_edges_1d(image,1.,0.)
        actual=find_edges_1d(image,1.,0.)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('eps',[1e-3,5e-4])
@pytest.mark.parametrize('k0',[cpu.K0,0.8*cpu.K0])
@pytest.mark.parametrize('mask_kind',['gray','asymmetric'])
def test_batched_jacobian_matches_original_central_difference(eps,k0,mask_kind):
    mask=np.full(32,.5) if mask_kind=='gray' else np.random.default_rng(31).random(32)
    src=([-2,0,1],np.array([1.,3.,2.]))
    expected=cpu.jacobian_1d(mask,8.,.6,k0=k0,src=src,eps=eps)
    actual=Imaging('cpu').jacobian(mask,8.,.6,k0=k0,src=src,eps=eps)
    np.testing.assert_allclose(actual,expected,atol=1e-9,rtol=1e-8)


@pytest.mark.parametrize('threshold',[0.,.5,1.])
@pytest.mark.parametrize('tolerance',[0.,.25,1.])
def test_batched_process_window_preserves_threshold_and_boundary_cases(threshold,tolerance):
    import types
    image=np.array([0.,0.,.5,1.,1.,.5,0.])
    engine=Imaging('cpu')
    def aerial(mask,dx,sigma,defocus=0.,source=None,dose=1.):
        return dose*np.roll(image,int(defocus))
    engine.aerial=aerial
    scope=dict(vars(cpu));scope['aerial_1d']=aerial
    scalar=types.FunctionType(cpu.process_window_1d.__code__,scope,
                              cpu.process_window_1d.__name__,cpu.process_window_1d.__defaults__)
    args=(image,1.,.6,threshold,[2.,5.],(1.,6.),tolerance,([0],[1.]))
    options=dict(doses=[.5,1.,1.5],focuses=[-1.,0.,1.])
    assert engine.process_window(*args,**options)==scalar(*args,**options)
