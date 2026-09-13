import numpy as np
import pytest
from physics_engine.litho import litho_aerial_image_scalar as cpu
from physics_engine.litho import litho_depth_of_focus_cert as fd
from physics_engine.litho.litho_aerial_image_scalar_gpu import aerial_contrast, aerial_image
from physics_engine.litho.litho_depth_of_focus_cert_gpu import aerial_contrast_defocus

@pytest.mark.parametrize('k1,sigma,dipole',[(.25,1.,True),(.5,0.,False),(.35,.5,False)])
def test_pupil_cutoff_and_binary_mask_grid(k1,sigma,dipole):
    expected=cpu.aerial_contrast(k1,.33,13.5e-9,sigma,dipole=dipole)
    actual=aerial_contrast(k1,.33,13.5e-9,sigma,dipole=dipole,device='cpu')
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)

@pytest.mark.parametrize('focus',[-1e-7,0.,1e-7])
def test_defocus_phase_units_and_symmetry(focus):
    expected=fd.aerial_contrast_defocus(.6,.55,13.5e-9,.5,focus)
    actual=aerial_contrast_defocus(.6,.55,13.5e-9,.5,focus,device='cpu')
    np.testing.assert_allclose(actual,expected,atol=1e-12,rtol=1e-10)

def test_bad_grid_refused():
    with pytest.raises(ValueError):aerial_image(.6,.33,13.5e-9,.5,npts=1,device='cpu')
