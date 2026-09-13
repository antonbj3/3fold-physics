"""Batched complex128 wavepacket autocorrelation on GPU; original time grids/gates."""
import types
import numpy as np
import torch
from physics_engine.quantum import quantum_revival as cpu


def autocorr(t,E1=np.pi**2/cpu.L0**2,anharm=0.,device='cuda',*,_spectrum=None):
    times=torch.as_tensor(np.asarray(t),dtype=torch.float64,device=device)
    if _spectrum is None:
        energy=torch.as_tensor(cpu.NN**2*E1*(1.+anharm*cpu.NN),device=device)
        weights=torch.as_tensor(cpu.CN2,device=device)
    else:
        energy,weights=_spectrum._arrays(E1,anharm,device)
    if _spectrum is not None and times.is_cuda and times.ndim==0 and _spectrum.graph_scalars:
        return _spectrum._scalar(times,energy,weights)
    phase=torch.exp(-1j*energy*times[...,None])
    return (torch.abs((weights*phase).sum(dim=-1))**2).cpu().numpy()



class AutocorrelationSeries:
    """Own one uploaded spectrum for sequential calls; never cache returned values.

    Level numbers and weights are copied at construction. Changing energy scale
    or detuning updates the sole energy buffer. Instances are not concurrent.
    """
    def __init__(self, device='cuda', graph_scalars=True):
        self.graph_scalars=graph_scalars
        self._graph=None
        self.device=torch.device(device)
        self._levels=cpu.NN.copy()
        self._weights=torch.tensor(cpu.CN2.copy(),device=self.device)
        self._key=None
        self._energy=None

    def _arrays(self,E1,anharm,device):
        if torch.device(device)!=self.device:
            raise ValueError('spectrum device must match the call device')
        key=(E1,anharm)
        if key!=self._key:
            values=torch.as_tensor(self._levels**2*E1*(1.+anharm*self._levels))
            if self._energy is None:
                self._energy=values.to(self.device)
            else:
                self._energy.copy_(values)
            self._key=key
        return self._energy,self._weights

    def _scalar(self,times,energy,weights):
        if self._graph is None:
            self._graph_times=torch.empty_like(times)
            self._graph_times.copy_(times)
            def evaluate():
                phase=torch.exp(-1j*energy*self._graph_times[...,None])
                return torch.abs((weights*phase).sum(dim=-1))**2
            stream=torch.cuda.Stream(device=times.device)
            stream.wait_stream(torch.cuda.current_stream(times.device))
            with torch.cuda.stream(stream):
                for _ in range(2):evaluate()
            torch.cuda.current_stream(times.device).wait_stream(stream)
            self._graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self._graph,stream=stream):
                self._graph_result=evaluate()
        self._graph_times.copy_(times)
        self._graph.replay()
        return self._graph_result.cpu().numpy()

    def __call__(self,t,E1=np.pi**2/cpu.L0**2,anharm=0.,device=None):
        return autocorr(t,E1,anharm,self.device if device is None else device,_spectrum=self)


def revival_time(L=cpu.L0,device='cuda'):
    E1=np.pi**2/L**2;Trev=2*np.pi/E1
    ts=np.linspace(.85*Trev,1.15*Trev,121);A=autocorr(ts,E1,device=device)
    i=int(np.argmax(A))
    if 0<i<len(ts)-1:
        y0,y1,y2=A[i-1],A[i],A[i+1]
        di=.5*(y0-y2)/(y0-2*y1+y2)
        return float(ts[i]+di*(ts[1]-ts[0]))
    return float(ts[i])


def revival_quality(L=cpu.L0,anharm=0.,device='cuda'):
    E1=np.pi**2/L**2;Trev=2*np.pi/E1
    ts=np.linspace(.7*Trev,1.15*Trev,6000)
    return float(autocorr(ts,E1,anharm,device).max())


def main(device='cuda'):
    series=AutocorrelationSeries(device)
    helper_scope=dict(globals(),autocorr=series)
    time_fn=types.FunctionType(revival_time.__code__,helper_scope,revival_time.__name__,revival_time.__defaults__)
    quality_fn=types.FunctionType(revival_quality.__code__,helper_scope,revival_quality.__name__,revival_quality.__defaults__)
    scope=dict(vars(cpu))
    scope['autocorr']=series
    scope['revival_time']=lambda *a,**kw:time_fn(*a,**kw,device=device)
    scope['revival_quality']=lambda *a,**kw:quality_fn(*a,**kw,device=device)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
