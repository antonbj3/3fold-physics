import os
import pytest
import numpy as np
from physics_engine.wave_optics import wave_optics_cell as cpu
from physics_engine.wave_optics import wave_optics_cell_gpu as gpu

def test_asymmetric_aperture_complex_phase_and_axes():
    aperture=np.zeros(127);aperture[19:47]=1;aperture[70:83]=.3
    for a,b in zip(cpu.fraunhofer_1d(aperture,1e-6,532e-9),gpu.fraunhofer_1d(aperture,1e-6,532e-9,'cpu')):
        np.testing.assert_allclose(a,b,atol=1e-15,rtol=1e-10)

def test_oblique_order_acceptance_and_ripple_field():
    x=np.linspace(-11e-6,11e-6,151)
    args=(x,22e-6,.9,532e-9,.4,.06)
    a,ma=cpu._coherent_image_intensity(*args);b,mb=gpu._coherent_image_intensity(*args,device='cpu')
    np.testing.assert_array_equal(ma,mb)
    np.testing.assert_allclose(a,b,atol=1e-12,rtol=1e-10)

@pytest.mark.parametrize('scale',[None,.125,1.75e-12])
def test_two_dimensional_phase_and_fft_shift(scale):
    aperture=np.random.default_rng(17).random((31,27))
    expected=np.fft.fftshift(np.fft.fft2(aperture))
    if scale is not None:expected=expected*scale
    np.testing.assert_allclose(gpu._fft2_shift(aperture,os.environ.get('PHYSICS_TEST_DEVICE','cpu'),scale=scale),expected,atol=1e-12,rtol=1e-10)


@pytest.mark.parametrize('n,D,window_factor',[(127,50e-6,4),(128,2.,4),(257,31e-6,7)])
def test_circular_aperture_boundary_mask_and_gate(monkeypatch,n,D,window_factor):
    masks=[];fft2=np.fft.fft2;gpu_fft=gpu._fft2_shift
    def cpu_fft(aperture,*a,**kw):
        masks.append(aperture.copy())
        return fft2(aperture,*a,**kw)
    def observed_gpu_fft(aperture,device='cuda',**kw):
        masks.append(aperture.detach().cpu().numpy().copy())
        return gpu_fft(aperture,device,**kw)
    monkeypatch.setattr(np.fft,'fft2',cpu_fft)
    monkeypatch.setattr(gpu,'_fft2_shift',observed_gpu_fft)
    expected=cpu.gate_G3_circular_aperture(D=D,n=n,window_factor=window_factor)
    actual=gpu.gate_G3_circular_aperture(D=D,n=n,window_factor=window_factor,device=os.environ.get('PHYSICS_TEST_DEVICE','cpu'))
    np.testing.assert_array_equal(masks[0],masks[1])
    assert expected==actual
