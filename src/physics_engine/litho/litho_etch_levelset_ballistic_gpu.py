"""Warp float64 etch time loops, parallel over trench widths.

The original explicit Euler update and full height/rate histories are retained.
The original VOF cross-check uses the geometric GPU sibling. Small two-trench
own gates and larger width-sweep timings are reported separately.
"""
import types
import numpy as np
import warp as wp
from physics_engine.litho import litho_etch_levelset_ballistic as cpu
from physics_engine.litho.vof_zalesak_cert_gpu import do_sweep
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
D=wp.float64

@wp.kernel
def etch_kernel(widths:wp.array(dtype=D),v0:D,dt:D,nsteps:int,hs:wp.array2d(dtype=D),rates:wp.array2d(dtype=D)):
    j=wp.tid();w=widths[j];h=D(0);hs[j,0]=D(0);rates[j,0]=v0
    for step in range(nsteps):
        ar=wp.max(h/w,D(0));r=v0/wp.sqrt(D(1)+D(4)*ar*ar)
        h=h+r*dt;hs[j,step+1]=h;rates[j,step+1]=r


def etch_batch(widths,v0=1.,dt=.02,nsteps=400,device='cuda'):
    widths=np.asarray(widths,dtype=float)
    if widths.ndim!=1 or not len(widths) or not np.isfinite(widths).all() or np.any(widths<=0):
        raise ValueError('positive finite trench widths required')
    if not np.isfinite([v0,dt]).all() or v0<0 or dt<=0 or not isinstance(nsteps,int) or nsteps<0:
        raise ValueError('nonnegative speed/steps and positive finite timestep required')
    w=wp.array(widths,dtype=D,device=device)
    hs=wp.empty((len(widths),nsteps+1),dtype=D,device=device);rates=wp.empty_like(hs)
    wp.launch(etch_kernel,len(widths),inputs=[w,D(v0),D(dt),nsteps,hs,rates],device=device)
    return np.arange(nsteps+1)*dt,hs.numpy(),rates.numpy()


def etch_levelset(w,v0=1.,dt=.02,nsteps=400,device='cuda'):
    t,h,r=etch_batch([w],v0,dt,nsteps,device)
    return t,h[0],r[0]


def main(device='cuda'):
    scope=dict(vars(cpu))
    scope['etch_levelset']=lambda *a,**kw:etch_levelset(*a,**kw,device=device)
    scope['a1_do_sweep']=lambda *a,**kw:do_sweep(*a,**kw,device=device)
    scope['a1_front_kinematics_check']=types.FunctionType(cpu.a1_front_kinematics_check.__code__,scope,
        cpu.a1_front_kinematics_check.__name__,cpu.a1_front_kinematics_check.__defaults__)
    return types.FunctionType(cpu.main.__code__,scope,cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
