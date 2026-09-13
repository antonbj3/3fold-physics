"""Cell-centered equal-area process chain; frozen constitutive arithmetic.

The41-fiber endpoint and midpoint experiments remain separate failed controls.
Default123 fibers must pass the declared ensemble refinement against369.
"""
from pathlib import Path
import sys
import numpy as np
HERE=Path(__file__).resolve().parent
if str(HERE) not in sys.path:sys.path.insert(0,str(HERE))
from physics_engine.chains import process_chain_v1 as frozen
mechanics=frozen.mechanics
fatigue=frozen.fatigue
PARAMS=frozen.PARAMS
digest=frozen.digest
evaluation=frozen.evaluation


def fiber_centers(count):
    if not isinstance(count,int) or isinstance(count,bool) or count<3:
        raise ValueError('At least three integer fibers required')
    return -.0004+(np.arange(count)+.5)*(.0008/count)


def thermal_field(samples=961,fibers=123):
    if not isinstance(samples,int) or isinstance(samples,bool) or samples<3:
        raise ValueError('At least three longitudinal samples required')
    y=fiber_centers(fibers)[None,:]
    x=np.linspace(.002,-.004,samples)[:,None]
    return np.concatenate([frozen.heat_rise(x[i:i+16],y,np.asarray(1e-5),**frozen.PARAMS) for i in range(0,samples,16)])


def temperatures(effective_rise,cooling):
    threshold=mechanics.MELTING-mechanics.AMBIENT
    sensible=np.where(effective_rise<=threshold,effective_rise,
                      np.maximum(threshold,effective_rise-270000/PARAMS['cp']))
    tail=sensible[-1][None,:]*np.linspace(1,0,cooling)[1:,None]
    return mechanics.AMBIENT+np.concatenate((np.zeros((1,effective_rise.shape[1])),sensible,tail))


def run_state(field,power=1.0,yield_scale=1.0,amplitude=100e6,cooling=513,relief=False):
    field=np.asarray(field,float)
    if field.ndim!=2 or min(field.shape)<3 or not np.isfinite(field).all():
        raise ValueError("Finite two-dimensional sampled thermal field required")
    count=field.shape[1]
    temp=temperatures(field*power,cooling)
    process,pdiag=mechanics.history(temp,yield_scale)
    residual=process[-1,:count]
    plastic=np.zeros(count) if relief else process[-1,count:2*count]
    loads=np.tile([amplitude,-amplitude],20)
    service,sdiag=mechanics.history(np.full((len(loads),count),mechanics.AMBIENT),yield_scale,plastic,loads)
    last=service[-2:,:count]
    low,high=last.min(axis=0),last.max(axis=0)
    plastic_increment=float(np.max(np.abs(np.diff(service[-3:,count:2*count],axis=0))))
    cycle_difference=float(np.max(np.abs(service[-2:,:count]-service[-4:-2,:count])))
    diag=dict(force_error=max(pdiag['force_error'],sdiag['force_error']),
              yield_excess=max(pdiag['yield_excess'],sdiag['yield_excess']),
              dissipation_min=min(pdiag['dissipation_min'],sdiag['dissipation_min']),
              liquid_stress=max(pdiag['liquid_stress'],sdiag['liquid_stress']),
              plastic_increment=plastic_increment,cycle_difference=cycle_difference,
              finite=bool(np.isfinite(process).all() and np.isfinite(service).all()))
    gates=dict(force=diag['force_error']<=1e-10,yield_bound=diag['yield_excess']<=1e-12,
               dissipation=diag['dissipation_min']>=-1e-12,liquid=diag['liquid_stress']==0,
               elastic_service=plastic_increment<=1e-12,stable_service=cycle_difference<=1e-10*mechanics.YIELD,
               finite=diag['finite'])
    return dict(low=low,high=high,residual=residual,gates=gates,diagnostics=diag,
                peak_temperature=float(temp.max()),molten_fibers=int(np.sum(temp.max(axis=0)>=mechanics.MELTING)),
                hashes=dict(temperature=digest(temp),process=digest(process),service=digest(service)),
                array_shapes=dict(temperature=list(temp.shape),process=list(process.shape),service=list(service.shape)))


if __name__=='__main__':
    import subprocess
    raise SystemExit(subprocess.run([sys.executable,str(HERE/'process_midpoint_certificate_v1.py')],check=False).returncode)
