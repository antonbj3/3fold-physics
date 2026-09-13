"""Torch float64 diagonalization of the original finite-difference Hamiltonian.

This dense GPU sibling computes the same stencil spectrum. It does not encode
analytic energies or change the CPU boundary convention. Dense eigensolving can
be slower than the original specialized CPU tridiagonal solver; report that cost.
"""
import types
import numpy as np
import torch
from physics_engine.quantum import schrodinger_1d as cpu


def solve(V,dx,k=6,device='cuda'):
    V=np.asarray(V,dtype=float)
    if V.ndim!=1 or not np.isfinite(V).all() or not np.isfinite(dx) or dx<=0 or not 1<=k<=len(V):
        raise ValueError('finite 1D potential, positive spacing and valid eigenpair count required')
    n=len(V)
    diagonal=torch.as_tensor(1./dx**2+V,device=device)
    off=torch.as_tensor(-.5/dx**2*np.ones(n-1),device=device)
    H=torch.diag(diagonal)+torch.diag(off,diagonal=1)+torch.diag(off,diagonal=-1)
    E,psi=torch.linalg.eigh(H)
    return E[:k].cpu().numpy(),(psi[:,:k]/np.sqrt(dx)).cpu().numpy()


def main(device='cuda'):
    scope=dict(vars(cpu));scope['solve']=lambda *a,**kw:solve(*a,**kw,device=device)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
