"""GPU Wannier-Stark eigensolve and full complex wavepacket propagation."""
import types
from collections import OrderedDict
import numpy as np
import torch
from physics_engine.quantum import bloch_oscillation as cpu


def bloch_amplitude(F=cpu.F0,Nsite=120,w=12.,nt=240,device='cuda',return_history=False,*,_operators=None):
    if not np.isfinite([F,w]).all() or F<=0 or w<=0 or Nsite<1 or nt<2:
        raise ValueError('positive force/width and valid space/time grids required')
    if _operators is None:
        n=torch.as_tensor(np.arange(-Nsite,Nsite+1).astype(float),device=device)
        size=len(n);off=torch.full((size-1,),-cpu.J,dtype=torch.float64,device=device)
        H=torch.diag(F*cpu.A*n)+torch.diag(off,diagonal=1)+torch.diag(off,diagonal=-1)
        E,V=torch.linalg.eigh(H)
    else:
        n,E,V=_operators._operator(F,Nsite,device)
    psi0=torch.exp(-n**2/(2*w**2));psi0=psi0/torch.linalg.vector_norm(psi0)
    c=V.T@psi0
    ts=torch.as_tensor(np.linspace(0.,2*np.pi/(F*cpu.A),nt),device=device)
    state=(V.to(torch.complex128)@(torch.exp(-1j*E[:,None]*ts[None,:])*c[:,None])).T
    xt=(n[None,:]*torch.abs(state)**2).sum(dim=1)
    amplitude=float((xt.max()-xt.min()).item())
    return (amplitude,state.cpu().numpy(),xt.cpu().numpy()) if return_history else amplitude



class BlochSeries:
    """Bounded operator reuse for sequential wavepackets; propagate every state anew.

    Width and time grids remain per-call inputs. A changed spatial grid discards
    the previous operators. Force, hopping and lattice spacing all enter the key.
    """
    def __init__(self,device='cuda',max_operators=4):
        if not isinstance(max_operators,int) or max_operators<1:
            raise ValueError('max_operators must be a positive integer')
        self.device=torch.device(device)
        self.max_operators=max_operators
        self._grid=None
        self._operators=OrderedDict()

    def _operator(self,F,Nsite,device):
        if torch.device(device)!=self.device:
            raise ValueError('operator device must match the call device')
        if self._grid!=Nsite:
            self._operators.clear()
            self._grid=Nsite
        key=(F,cpu.J,cpu.A)
        if key not in self._operators:
            n=torch.as_tensor(np.arange(-Nsite,Nsite+1).astype(float),device=self.device)
            size=len(n);off=torch.full((size-1,),-cpu.J,dtype=torch.float64,device=self.device)
            H=torch.diag(F*cpu.A*n)+torch.diag(off,diagonal=1)+torch.diag(off,diagonal=-1)
            E,V=torch.linalg.eigh(H)
            if len(self._operators)==self.max_operators:
                self._operators.popitem(last=False)
            self._operators[key]=(n,E,V)
        self._operators.move_to_end(key)
        return self._operators[key]

    def __call__(self,F=cpu.F0,Nsite=120,w=12.,nt=240,return_history=False):
        return bloch_amplitude(F,Nsite,w,nt,self.device,return_history,_operators=self)


def main(device='cuda'):
    scope=dict(vars(cpu));scope['bloch_amplitude']=BlochSeries(device)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
