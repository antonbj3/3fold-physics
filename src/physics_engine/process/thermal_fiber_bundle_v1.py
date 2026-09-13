"""Illustrative equal-area parallel fibers with melting and perfect plasticity."""
import hashlib
import numpy as np

AMBIENT=293.0
MELTING=1600.0
MODULUS=200e9
YIELD=600e6
EXPANSION=1.4e-5


def advance(temperature,plastic,mean_stress=0.0,yield_scale=1.0):
    temperature=np.asarray(temperature,float)
    plastic=np.asarray(plastic,float)
    if temperature.shape!=plastic.shape or temperature.ndim!=1:
        raise ValueError('Equal one-dimensional fiber arrays required')
    solid=np.clip((MELTING-temperature)/(MELTING-AMBIENT),0,1)
    e_mod=MODULUS*solid
    cap=YIELD*yield_scale*solid
    if not np.isfinite(temperature).all() or not np.isfinite(plastic).all() or not np.isfinite(mean_stress) or yield_scale<=0:
        raise ValueError('Finite material state required')
    if e_mod.sum()==0 or abs(mean_stress)>=cap.mean():
        raise ValueError('No load-bearing equilibrium')
    target=plastic+EXPANSION*(temperature-AMBIENT)
    live=e_mod>0
    width=YIELD*yield_scale/MODULUS
    lo=float(target[live].min()-2*width-abs(mean_stress)/e_mod[live].min())
    hi=float(target[live].max()+2*width+abs(mean_stress)/e_mod[live].min())
    common=float((np.sum(e_mod*target)+mean_stress*len(target))/e_mod.sum())
    for _ in range(80):
        trial=e_mod*(common-target)
        stress=np.clip(trial,-cap,cap)
        residual=float(stress.mean()-mean_stress)
        if abs(residual)<=1e-12*YIELD*yield_scale:
            break
        if residual>0:hi=common
        else:lo=common
        tangent=float(np.mean(np.where(np.abs(trial)<cap,e_mod,0)))
        candidate=common-residual/tangent if tangent>0 else float('nan')
        common=candidate if lo<candidate<hi else (lo+hi)/2
    else:raise RuntimeError('Force equilibrium did not converge')
    new=plastic.copy()
    yielding=live & (np.abs(trial)>cap)
    new[yielding]=common-EXPANSION*(temperature[yielding]-AMBIENT)-stress[yielding]/e_mod[yielding]
    # Liquid has no elastic stress; its reference follows current free strain.
    new[~live]=common-EXPANSION*(temperature[~live]-AMBIENT)
    dissipation=stress*(new-plastic)
    return stress,new,common,dict(force_error=abs(residual)/(YIELD*yield_scale),
                                 yield_excess=float(np.max(np.abs(stress)-cap))/(YIELD*yield_scale),
                                 dissipation_min=float(dissipation.min())/(YIELD*yield_scale),
                                 liquid_stress=float(np.max(np.abs(stress[~live]))) if (~live).any() else 0.0)


def history(temperatures,yield_scale=1.0,plastic=None,loads=None):
    temperatures=np.asarray(temperatures,float)
    plastic=np.zeros(temperatures.shape[1]) if plastic is None else np.array(plastic,copy=True)
    loads=np.zeros(len(temperatures)) if loads is None else np.asarray(loads,float)
    if loads.shape!=(len(temperatures),):raise ValueError('One mean load per sample required')
    rows=[]
    diagnostics=[]
    for temperature,load in zip(temperatures,loads):
        stress,plastic,common,d=advance(temperature,plastic,float(load),yield_scale)
        rows.append(np.concatenate((stress,plastic,[common])))
        diagnostics.append(d)
    return np.array(rows),dict(force_error=max(d['force_error'] for d in diagnostics),
                              yield_excess=max(d['yield_excess'] for d in diagnostics),
                              dissipation_min=min(d['dissipation_min'] for d in diagnostics),
                              liquid_stress=max(d['liquid_stress'] for d in diagnostics))


def controls():
    uniform=np.repeat(np.linspace(AMBIENT,1200,51)[:,None],5,axis=1)
    u,ud=history(uniform)
    # Two elastic fibers: equal modulus, zero net force, one prescribed rise.
    s,p,e,d=advance(np.array([AMBIENT,AMBIENT+10]),np.zeros(2))
    modulus=MODULUS*np.array([1,(MELTING-AMBIENT-10)/(MELTING-AMBIENT)])
    exact_strain=float(modulus[1]*EXPANSION*10/modulus.sum())
    exact=modulus*(exact_strain-np.array([0,EXPANSION*10]))
    liquid,_,_,ld=advance(np.array([AMBIENT,MELTING+100]),np.zeros(2))
    zero,zd=history(np.full((3,5),AMBIENT))
    gates=dict(uniform_free=bool(np.max(np.abs(u[:,:5]))<=1e-10*YIELD),
               two_fiber_elastic=bool(np.max(np.abs(s-exact))<=1e-10*YIELD and (p==0).all()),
               zero_heat=bool((zero==0).all()),liquid_zero=bool(liquid[1]==0),
               equilibrium=bool(max(ud['force_error'],d['force_error'],ld['force_error'],zd['force_error'])<=1e-10))
    return dict(gates=gates,control_hashes={k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in dict(uniform=u,elastic_stress=s,elastic_plastic=p,liquid=liquid,zero=zero).items()},uniform_max_stress=float(np.max(np.abs(u[:,:5]))),two_fiber_error=float(np.max(np.abs(s-exact))))


if __name__=='__main__':
    import json
    a,b=controls(),controls()
    gates=dict(a['gates'],exact_repeats=a==b)
    print(json.dumps(dict(legs=[a,b],gates=gates)),flush=True)
    raise SystemExit(0 if all(gates.values()) else 1)
