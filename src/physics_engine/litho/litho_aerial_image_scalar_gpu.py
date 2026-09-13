"""Float64 torch Abbe imaging, batched over the original source points.

Host grid/mask construction is retained to preserve binary mask decisions at
sampled cosine zeros. FFT, source propagation, intensity sum and contrast use the
requested device. The CPU main and all four gates are reused without mutation.
"""
import types
import numpy as np
import torch
from physics_engine.litho import litho_aerial_image_scalar as cpu


def aerial_image(k1, NA, lam, sigma, defocus=0., Nper=24, npts=8192,
                 dipole=False, device='cuda'):
    if not np.isfinite([k1,NA,lam,sigma,defocus,Nper]).all() or min(k1,NA,lam,Nper)<=0 or sigma<0:
        raise ValueError('finite positive geometry and nonnegative coherence required')
    if not isinstance(npts,int) or npts<2:
        raise ValueError('at least two integer samples required')
    p=2.*k1*lam/NA
    x=np.linspace(0.,Nper*p,npts,endpoint=False)
    dx=x[1]-x[0]
    mask=(np.cos(2.*np.pi*x/p)>=0.).astype(float)
    f=np.fft.fftfreq(npts,dx); fc=NA/lam
    sources=np.array([0.]) if sigma<=0 else (np.array([-sigma*fc,sigma*fc]) if dipole else np.linspace(-sigma*fc,sigma*fc,21))
    frequencies=torch.as_tensor(f[None,:]+sources[:,None],device=device)
    pupil=(torch.abs(frequencies)<=fc*(1.+1e-9))
    phase=torch.exp(1j*np.pi*defocus*lam*frequencies**2)
    spectrum=torch.fft.fft(torch.as_tensor(mask,device=device))
    field=torch.fft.ifft(spectrum[None,:]*pupil*phase,dim=-1)
    return (field.real**2+field.imag**2).mean(dim=0)


def aerial_contrast(k1, NA, lam, sigma, Nper=24, npts=8192, dipole=False, device='cuda'):
    image=aerial_image(k1,NA,lam,sigma,Nper=Nper,npts=npts,dipole=dipole,device=device)
    hi,lo=image.max(),image.min()
    return float(((hi-lo)/(hi+lo+1e-30)).item())


def main(device='cuda'):
    scope=dict(vars(cpu))
    scope['aerial_contrast']=lambda *a,**kw:aerial_contrast(*a,**kw,device=device)
    scope['resolution_k1']=types.FunctionType(cpu.resolution_k1.__code__,scope,
        cpu.resolution_k1.__name__,cpu.resolution_k1.__defaults__)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
