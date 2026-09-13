"""GPU Eagar-Tsai/Rosenthal spatial sweeps, original pool decisions and gates.

The finite-u quadrature, cutoffs, material parameters and scalar limit are
unchanged. The same spatial grids are evaluated in bounded float64 GPU tiles.
"""
import types
import numpy as np
import torch
from physics_engine.process import lpbf_meltpool_render_match as cpu


def dT_eagar_tsai(xi,y,z,P,v,eta,sigma_b,k,rho,cp,n_u=1500,device='cuda',tile=2048):
    if min(v,k,rho,cp)<=0 or sigma_b<0 or n_u<2 or tile<1:
        raise ValueError('positive material/speed, valid beam and quadrature required')
    alpha=k/(rho*cp);t0=sigma_b**2/(2.*alpha)
    u_max=np.sqrt(28.*alpha)/v+np.sqrt(8.*t0)
    u=torch.as_tensor(np.linspace(1e-7,u_max,n_u),device=device)
    tau=u**2
    coords=np.broadcast_arrays(xi,y,z);shape=coords[0].shape
    flat=[torch.as_tensor(np.array(a,copy=True).ravel(),dtype=torch.float64,device=device) for a in coords]
    out=torch.empty(flat[0].numel(),dtype=torch.float64,device=device)
    pref=2.*eta*P/(rho*cp*(4*np.pi*alpha)**1.5)
    for start in range(0,len(out),tile):
        X,Y,Z=[a[start:start+tile,None] for a in flat]
        lat=(X+v*tau)**2+Y**2
        integrand=(2./(tau+t0))*torch.exp(-lat/(4*alpha*(tau+t0))-Z**2/(4*alpha*tau))
        out[start:start+tile]=pref*torch.trapezoid(integrand,u,dim=-1)
    return out.reshape(shape).cpu().numpy()


def dT_rosenthal(xi,y,z,P,v,eta,k,rho,cp,device='cuda'):
    X,Y,Z=[torch.as_tensor(np.array(a,copy=True),dtype=torch.float64,device=device) for a in (xi,y,z)]
    alpha=k/(rho*cp);r=torch.sqrt(X**2+Y**2+Z**2)+1e-12
    return ((eta*P)/(2*np.pi*k*r)*torch.exp(-v*(X+r)/(2*alpha))).cpu().numpy()


def solver_scope(device='cuda'):
    scope=dict(vars(cpu))
    scope['dT_eagar_tsai']=lambda *a,**kw:dT_eagar_tsai(*a,**kw,device=device)
    scope['dT_rosenthal']=lambda *a,**kw:dT_rosenthal(*a,**kw,device=device)
    fn=cpu.pool_depth_width
    scope['pool_depth_width']=types.FunctionType(fn.__code__,scope,fn.__name__,fn.__defaults__)
    return scope


def main(device='cuda'):
    return types.FunctionType(cpu.main.__code__,solver_scope(device),cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
