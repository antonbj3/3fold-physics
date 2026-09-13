"""GPU Gaussian-source quadrature with original melt-width decisions/gates.

Time quadrature nodes and measured-power interpolation use the unchanged host
formulas. Radial Green-function integration is batched in torch float64. The
original latent-heat/convection omissions and conduction-only residual remain.
"""
import types
import numpy as np
import torch
from physics_engine.process import weld_goldak_spot_nist as cpu


def temp_field(t_out,r_grid,tP,P,mat,nu=1500,device='cuda',tile=8):
    t_out=np.asarray(t_out);r_grid=np.asarray(r_grid);tP=np.asarray(tP);P=np.asarray(P)
    if nu<2 or tile<1 or len(tP)<2 or np.any(np.diff(tP)<=0):
        raise ValueError('ordered power samples, positive tile and >=2 quadrature nodes required')
    if not all(np.isfinite(v).all() for v in (t_out,r_grid,tP,P)) or len(P)!=len(tP):
        raise ValueError('finite matching input arrays required')
    T=np.full((len(t_out),len(r_grid)),mat['T0'],dtype=float)
    alpha=mat['k']/(mat['rho']*mat['cp'])
    C=2./(mat['rho']*mat['cp']*np.sqrt(4.*np.pi*alpha))
    r=torch.as_tensor(r_grid,dtype=torch.float64,device=device)
    valid=np.where(np.minimum(t_out,tP[-1])>0)[0]
    for start in range(0,len(valid),tile):
        indices=valid[start:start+tile]
        nodes=np.array([np.linspace(0.,np.sqrt(min(t_out[i],tP[-1])),nu) for i in indices])
        power=np.array([np.interp(t_out[i]-u*u,tP,P) for i,u in zip(indices,nodes)])
        u=torch.as_tensor(nodes,device=device);pv=torch.as_tensor(power,device=device)
        s=4.*alpha*u*u+cpu.W_E2*cpu.W_E2/2.
        g=pv[:,:,None]*torch.exp(-r[None,None,:]**2/s[:,:,None])/(np.pi*s[:,:,None])
        T[indices]+= (C*2.*torch.trapezoid(g,u[:,:,None],dim=1)).cpu().numpy()
    return T


def solver_scope(device='cuda',data_dir=None):
    scope=dict(vars(cpu))
    if data_dir is not None:scope['D']=str(data_dir)
    scope['temp_field']=lambda *a,**kw:temp_field(*a,**kw,device=device)
    for name in ('width_series','load_tda','load_tdw','_synthetic_tdw'):
        fn=getattr(cpu,name);scope[name]=types.FunctionType(fn.__code__,scope,name,fn.__defaults__)
    return scope


def main(device='cuda'):
    return types.FunctionType(cpu.main.__code__,solver_scope(device),cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
