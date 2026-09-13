"""Moving Gaussian heat integral with a downstream-aware time horizon.

Same half-space Green function as the frozen thermal reference. This is a
constant-property effective-enthalpy approximation, not a flow or vapor model.
"""
import numpy as np


def heat_rise(xi,y,z,P,v,eta,sigma_b,k,rho,cp,n_u=3000,horizon_scale=1.0):
    if min(v,sigma_b,k,rho,cp)<=0 or P<0 or not 0<=eta<=1:
        raise ValueError('Invalid thermal parameters')
    xi,y,z=np.broadcast_arrays(np.asarray(xi,float),np.asarray(y,float),np.asarray(z,float))
    if not all(np.isfinite(a).all() for a in (xi,y,z)) or (z<0).any():
        raise ValueError('Finite half-space coordinates required')
    alpha=k/(rho*cp)
    t0=sigma_b**2/(2*alpha)
    upper=(np.sqrt(np.maximum(-xi,0)/v)+np.sqrt(28*alpha)/v+np.sqrt(8*t0))*horizon_scale
    # Row-specific intervals preserve resolution near the source while adding
    # the retarded arrival time for coordinates downstream.
    s=np.linspace(0,1,n_u)
    u=1e-7+(upper[...,None]-1e-7)*s
    tau=u*u
    integrand=2/(tau+t0)*np.exp(-((xi[...,None]+v*tau)**2+y[...,None]**2)/(4*alpha*(tau+t0))-z[...,None]**2/(4*alpha*tau))
    return 2*eta*P/(rho*cp*(4*np.pi*alpha)**1.5)*np.trapezoid(integrand,u,axis=-1)


if __name__=='__main__':
    from pathlib import Path
    import subprocess
    import sys
    probe=Path(__file__).resolve().parents[1]/'chains/process_heat_horizon_probe.py'
    raise SystemExit(subprocess.run([sys.executable,str(probe)],check=False).returncode)
