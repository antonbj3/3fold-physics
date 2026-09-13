"""Warp checkerboard Metropolis with the CPU sibling's exact random stream.

Host NumPy supplies the same initial spins and all half-sweep random draws,
including draws for inactive sites. Host exp probabilities preserve acceptance
thresholds exactly; GPU integer stencil updates and observations are exact.
Full spin histories are allocated only when return_history=True.
Random draws are transferred in bounded sweep blocks without changing draw order.
Ping-pong fields retain the CPU simultaneous-half-sweep semantics even for odd L.
"""
import types
import numpy as np
import warp as wp
from physics_engine.quantum import ising_2d as cpu
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})

@wp.kernel
def half_sweep(s:wp.array2d(dtype=wp.int32),random:wp.array3d(dtype=wp.float64),prob:wp.array(dtype=wp.float64),step:int,parity:int,out:wp.array2d(dtype=wp.int32)):
    i,j=wp.tid();n=s.shape[0];value=s[i,j]
    if (i+j)%2==parity:
        nb=s[(i+1)%n,j]+s[(i+n-1)%n,j]+s[i,(j+1)%n]+s[i,(j+n-1)%n]
        energy=2*value*nb
        if energy<=0 or random[step,i,j]<prob[(energy+8)//4]:value=-value
    out[i,j]=value

@wp.kernel
def observe(s:wp.array2d(dtype=wp.int32),step:int,M:wp.array(dtype=wp.int32),E:wp.array(dtype=wp.int32),history:wp.array3d(dtype=wp.int32),store_history:int):
    i,j=wp.tid();n=s.shape[0]
    wp.atomic_add(M,step,s[i,j])
    wp.atomic_add(E,step,-s[i,j]*(s[(i+n-1)%n,j]+s[i,(j+n-1)%n]))
    if store_history!=0:history[step,i,j]=s[i,j]


@wp.kernel
def observe_offset(s:wp.array2d(dtype=wp.int32),local_step:int,start:wp.array(dtype=wp.int32),M:wp.array(dtype=wp.int32),E:wp.array(dtype=wp.int32),history:wp.array3d(dtype=wp.int32),store_history:int):
    i,j=wp.tid();n=s.shape[0];step=start[0]+local_step
    wp.atomic_add(M,step,s[i,j])
    wp.atomic_add(E,step,-s[i,j]*(s[(i+n-1)%n,j]+s[i,(j+n-1)%n]))
    if store_history!=0:history[step,i,j]=s[i,j]


def graph_sweeps(state,out,prob,M,E,history,rng,L,eq,meas,batch,store_history,device,buffers):
    if "random" not in buffers:
        buffers["random"]=wp.empty((2*min(batch,eq+meas),L,L),dtype=wp.float64,device=device)
        buffers["offset"]=wp.zeros(1,dtype=wp.int32,device=device)
        buffers["graphs"]={}
    random=buffers["random"];offset=buffers["offset"];graphs=buffers["graphs"];start=0
    while start<eq+meas:
        measuring=start>=eq
        count=min(batch,eq+meas-start,eq-start if not measuring else meas)
        draws=rng.random((2*count,L,L))
        host=wp.array(draws,dtype=wp.float64,device='cpu',copy=False)
        wp.copy(random,host,count=draws.size)
        if measuring:offset.assign(np.array([start-eq],dtype=np.int32))
        key=(count,measuring)
        if key not in graphs:
            with wp.ScopedCapture(device=device,force_module_load=True) as capture:
                for step in range(count):
                    for parity in (0,1):
                        wp.launch(half_sweep,(L,L),inputs=[state,random,prob,2*step+parity,parity,out],device=device)
                        state,out=out,state
                    if measuring:wp.launch(observe_offset,(L,L),inputs=[state,step,offset,M,E,history,int(store_history)],device=device)
            graphs[key]=capture.graph
        wp.capture_launch(graphs[key])
        wp.synchronize_device(device)
        del host,draws
        start+=count


def _simulate(L,T,rng,eq,meas,device,return_history,random_batch_sweeps,buffers):
    if not isinstance(L,int) or L<2 or not np.isfinite(T) or T<=0 or eq<0 or meas<1:
        raise ValueError('valid lattice, positive finite temperature and measurement count required')
    if not isinstance(random_batch_sweeps,int) or isinstance(random_batch_sweeps,bool) or random_batch_sweeps<1:
        raise ValueError('random_batch_sweeps must be a positive integer')
    initial=rng.choice([-1,1],size=(L,L))
    probabilities=np.exp(-np.arange(-8,9,4,dtype=float)/T)
    if "state" not in buffers:
        buffers["state"]=wp.empty((L,L),dtype=wp.int32,device=device)
        buffers["out"]=wp.empty_like(buffers["state"])
        buffers["prob"]=wp.empty(5,dtype=wp.float64,device=device)
        buffers["M"]=wp.empty(meas,dtype=wp.int32,device=device)
        buffers["E"]=wp.empty_like(buffers["M"])
        buffers["history"]=wp.empty((meas,L,L) if return_history else (1,1,1),dtype=wp.int32,device=device)
    state,out,prob,M,E,history=(buffers[k] for k in ("state","out","prob","M","E","history"))
    state.assign(initial.astype(np.int32));prob.assign(probabilities);M.zero_();E.zero_()
    if wp.get_device(device).is_cuda:
        graph_sweeps(state,out,prob,M,E,history,rng,L,eq,meas,random_batch_sweeps,return_history,device,buffers)
    else:
        for start in range(0,eq+meas,random_batch_sweeps):
            count=min(random_batch_sweeps,eq+meas-start)
            draws=rng.random((2*count,L,L))
            random=wp.array(draws,dtype=wp.float64,device=device)
            for local_step in range(count):
                step=start+local_step
                for parity in (0,1):
                    wp.launch(half_sweep,(L,L),inputs=[state,random,prob,2*local_step+parity,parity,out],device=device)
                    state,out=out,state
                if step>=eq:wp.launch(observe,(L,L),inputs=[state,step-eq,M,E,history,int(return_history)],device=device)
            # Complete all readers before releasing this block and allocating its successor.
            wp.synchronize_device(device)
            del random,draws
    Ms=np.abs(M.numpy().astype(float)/(L*L));Es=E.numpy().astype(float)/(L*L)
    chi=(np.mean(Ms**2)-np.mean(Ms)**2)*L*L/T
    result=(float(Ms.mean()),float(chi),float(Es.mean()))
    return (result,history.numpy() if wp.get_device(device).is_cuda else history.numpy().copy(),Ms,Es) if return_history else result


def simulate(L,T,rng,eq=400,meas=600,device='cuda',return_history=False,random_batch_sweeps=32):
    return _simulate(L,T,rng,eq,meas,device,return_history,random_batch_sweeps,{})


class SimulationSeries:
    """Own buffers and graphs for sequential simulations on one device.

    Each call resets spins and observations and consumes the supplied RNG anew.
    Changing geometry, counts or storage mode releases the previous layout.
    An instance must not be called concurrently.
    """
    def __init__(self,device='cuda',return_history=False,random_batch_sweeps=32):
        self.device=device;self.return_history=return_history
        self.random_batch_sweeps=random_batch_sweeps;self._layout=None;self._buffers={}

    def __call__(self,L,T,rng,eq=400,meas=600):
        layout=(L,eq,meas,self.return_history,self.random_batch_sweeps,self.device)
        if layout!=self._layout:
            self._buffers={};self._layout=layout
        return _simulate(L,T,rng,eq,meas,self.device,self.return_history,self.random_batch_sweeps,self._buffers)


def main(device='cuda'):
    scope=dict(vars(cpu));scope['simulate']=SimulationSeries(device=device)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
