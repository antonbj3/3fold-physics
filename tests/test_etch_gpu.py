import numpy as np
import pytest
from physics_engine.litho import litho_etch_levelset_ballistic as cpu
from physics_engine.litho.litho_etch_levelset_ballistic_gpu import etch_batch

@pytest.mark.parametrize('steps',[0,1,400])
def test_full_euler_histories_and_first_rate(steps):
    widths=[.1,1.,4.,10.]
    t,h,r=etch_batch(widths,nsteps=steps,device='cpu')
    for j,w in enumerate(widths):
        ct,ch,cr=cpu.etch_levelset(w,nsteps=steps)
        np.testing.assert_array_equal(t,ct)
        np.testing.assert_allclose(h[j],ch,atol=1e-12,rtol=1e-10)
        np.testing.assert_allclose(r[j],cr,atol=1e-12,rtol=1e-10)

def test_nonphysical_width_refused():
    with pytest.raises(ValueError):etch_batch([0.],device='cpu')
