import numpy as np
import pytest
from physics_engine.litho import vof_zalesak_cert as cpu
from physics_engine.litho import vof_zalesak_cert_gpu as gpu

@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('scheme',['geom','upwind'])
def test_face_donors_normals_boundaries_and_compression(axis,scheme):
    rng=np.random.default_rng(17);n=11;F=rng.random((n,n))
    F[0]=0;F[-1]=1;F[3,4]=1e-6;F[6,5]=1-1e-6
    c=rng.uniform(-.4,.4,size=(n+1,n) if axis==0 else (n,n+1))
    fn=cpu.do_sweep if scheme=='geom' else cpu.upwind_sweep
    expected=fn(F,c,axis);actual=gpu.do_sweep(F,c,axis,scheme,device='cpu')
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)

def test_invalid_courant_refused():
    with pytest.raises(ValueError):gpu.do_sweep(np.ones((3,3)),np.ones((4,3))*2,0,device='cpu')
