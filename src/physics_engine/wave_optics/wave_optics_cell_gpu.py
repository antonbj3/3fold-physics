"""GPU Fourier propagation with unchanged CPU instrument/ripple gates.

Masks, order acceptance and observation grids remain the original host formulas.
FFT/FFT2 and coherent Fourier-series propagation run in torch complex128.
"""
import types
import numpy as np
import torch
from physics_engine.wave_optics import wave_optics_cell as cpu
_first_zero_sintheta=cpu._first_zero_sintheta


def fraunhofer_1d(aperture,dx,wavelength,device='cuda'):
    n=len(aperture)
    frequencies=np.fft.fftshift(np.fft.fftfreq(n,d=dx))
    field=torch.fft.fftshift(torch.fft.fft(torch.as_tensor(aperture,dtype=torch.float64,device=device)))*dx
    return wavelength*frequencies,field.cpu().numpy()


def _fft2_shift(aperture,device='cuda',scale=None):
    field=torch.fft.fftshift(torch.fft.fft2(torch.as_tensor(aperture,dtype=torch.float64,device=device)))
    if scale is not None:field=field*scale
    return field.cpu().numpy()


def _coherent_image_intensity(x,pitch,fill_factor,wavelength,na_obj,sin_theta_i,m_cutoff=200,device='cuda'):
    m=np.arange(-m_cutoff,m_cutoff+1)
    passed=np.abs(sin_theta_i+m*wavelength/pitch)<=na_obj
    modes=m[passed]
    if not len(modes):return np.zeros_like(x)
    coefficients=np.array([cpu.grating_fourier_coeff(mm,fill_factor) for mm in modes])
    X=torch.as_tensor(x,dtype=torch.float64,device=device)
    M=torch.as_tensor(modes,dtype=torch.float64,device=device)
    phase=torch.exp(1j*2*np.pi*torch.outer(X,M)/pitch)
    field=phase@torch.as_tensor(coefficients,dtype=torch.complex128,device=device)
    return (torch.abs(field)**2).cpu().numpy(),modes


def gate_G3_circular_aperture(D=50e-6, wavelength=532e-9, n=4096, window_factor=60, device="cuda"):
    """G3: circular aperture diameter D, first Airy dark ring at sin(theta)=1.22*lambda/D."""
    x_max = window_factor * D
    dx = 2 * x_max / n
    coords = (np.arange(n) - n // 2) * dx
    grid = torch.as_tensor(coords, dtype=torch.float64, device=device)
    R = torch.sqrt(grid[None, :] ** 2 + grid[:, None] ** 2)
    aperture = (R <= D / 2).to(torch.float64)
    field = _fft2_shift(aperture, device, scale=dx ** 2)
    freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    sin_theta_axis = wavelength * freqs
    # radial profile through the center row (by symmetry, radial = axial cut)
    center = n // 2
    row = np.abs(field[center, :]) ** 2
    sin_theta_row = sin_theta_axis
    analytic = 1.22 * wavelength / D
    # pass the FULL symmetric array (not pre-sliced) — _first_zero_sintheta locates center=len//2 itself
    measured = _first_zero_sintheta(sin_theta_row, row, exclude_radius=0.3 * analytic)
    rel_err = abs(measured - analytic) / analytic
    grid_angular_resolution = wavelength * (1.0 / (n * dx))   # sin_theta bin size — the honest resolution note
    return dict(gate="G3_circular_aperture", D=D, wavelength=wavelength,
                analytic_sin_theta=analytic, measured_sin_theta=measured,
                grid_sin_theta_resolution=grid_angular_resolution,
                rel_err_pct=100 * rel_err, pass_=rel_err < 0.5)

def run_all(device='cuda'):
    scope=dict(vars(cpu))
    scope['fraunhofer_1d']=lambda *a,**kw:fraunhofer_1d(*a,**kw,device=device)
    scope['_coherent_image_intensity']=lambda *a,**kw:_coherent_image_intensity(*a,**kw,device=device)
    scope['gate_G3_circular_aperture']=lambda *a,**kw:gate_G3_circular_aperture(*a,**kw,device=device)
    for name in ('gate_G1_single_slit','gate_G2_double_slit','e13_ripple','run_all'):
        fn=getattr(cpu,name);scope[name]=types.FunctionType(fn.__code__,scope,name,fn.__defaults__)
    return scope['run_all']()

if __name__=='__main__':
    results=run_all()
    for result in results:print(result)
    raise SystemExit(0 if all(g['pass_'] for g in results[:3]) else 1)
