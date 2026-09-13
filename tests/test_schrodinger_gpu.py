import numpy as np
import pytest
from physics_engine.quantum import schrodinger_1d as cpu
from physics_engine.quantum import schrodinger_1d_gpu as gpu

@pytest.mark.parametrize('n',[31,64])
def test_asymmetric_potential_eigenpairs_and_stencil_residual(n):
    x=np.linspace(-3,3,n);dx=x[1]-x[0];V=.5*x*x+.12*np.sin(1.7*x)
    E,P=cpu.solve(V,dx,5);G,Q=gpu.solve(V,dx,5,device='cpu')
    signs=np.where(np.sum(P*Q,axis=0)<0,-1.,1.);Q*=signs
    np.testing.assert_allclose(G,E,atol=1e-10,rtol=1e-9)
    np.testing.assert_allclose(Q,P,atol=1e-10,rtol=1e-9)
    H=np.diag(1/dx**2+V)+np.diag(np.full(n-1,-.5/dx**2),1)+np.diag(np.full(n-1,-.5/dx**2),-1)
    np.testing.assert_allclose(H@Q,Q*G,atol=1e-10,rtol=1e-9)

def test_bad_spacing_refused():
    with pytest.raises(ValueError):gpu.solve(np.ones(10),0.,device='cpu')
