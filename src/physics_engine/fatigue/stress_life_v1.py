"""Declared stress-life equation; numerical certification, no material calibration.

Conventions: https://doc.comsol.com/6.3/doc/com.comsol.help.fatigue/fatigue_ug_sme.4.20.html
Mean correction: https://doc.comsol.com/6.4/doc/com.comsol.help.fatigue/fatigue_ug_sme.4.28.html
"""
import hashlib
import numpy as np


def damage_per_cycle(low,high,strength=1.5e9,ultimate=1.2e9,exponent=-.1):
    low,high=np.broadcast_arrays(np.asarray(low,float),np.asarray(high,float))
    if not all(np.isfinite(x).all() for x in (low,high)) or (high<low).any():
        raise ValueError('Finite ordered stress extrema required')
    if not np.isfinite([strength,ultimate,exponent]).all() or min(strength,ultimate)<=0 or exponent>=0:
        raise ValueError('Positive strengths and negative exponent required')
    mean=(high+low)/2
    amplitude=(high-low)/2
    if (mean>=ultimate).any() or (high>=ultimate).any():
        raise ValueError('Stress reaches ultimate strength')
    effective=amplitude/(1-mean/ultimate)
    if (effective>=strength).any():raise ValueError('Outside the declared stress-life domain')
    return 2*(effective/strength)**(-1/exponent)


def controls():
    cycles=np.array([1e4,1e5,1e6,1e7])
    strength=1.5e9
    amplitude=strength*(2*cycles)**(-.1)
    d=damage_per_cycle(-amplitude,amplitude)
    error=float(np.max(np.abs(d*cycles-1)))
    invalid=False
    try:damage_per_cycle(1.2e9,1.2e9)
    except ValueError:invalid=True
    base=damage_per_cycle(-100e6,100e6)
    tensile=damage_per_cycle(0,200e6)
    return dict(gates=dict(inverse_law=error<=1e-12,zero_amplitude=bool(damage_per_cycle(0,0)==0),
                           tensile_increases=bool(tensile>base),invalid_rejected=invalid),inverse_relative_error=error,control_hashes={k:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for k,v in dict(inverse_damage=d,cycles=cycles,amplitude=amplitude,base=base,tensile=tensile).items()})


if __name__=='__main__':
    import json
    a,b=controls(),controls()
    gates=dict(a['gates'],exact_repeats=a==b)
    print(json.dumps(dict(legs=[a,b],gates=gates)),flush=True)
    raise SystemExit(0 if all(gates.values()) else 1)
