"""Measure added radius/index coordinates using only frozen optical instruments."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import lens_design_search_v1 as baseline

lrc, lrg = baseline.lrc, baseline.lrg
FREE = (2,3,7,8,9,10,11,12,13,14,15)
ADDED = (2,3,7,8,10,11,12,14,15)
INDEX_CHOICES = (1.48914,1.51872,1.64128,1.81265)
PATENT = np.array([baseline.PATENT_SPACE1,baseline.PATENT_SPACE2,*[lrc.P[i][0] for i in FREE]],np.float64)


def evaluate(vectors, ax, ay, index=None):
    vectors = np.atleast_2d(vectors)
    rows = np.column_stack([vectors[:,:2],np.full(len(vectors),baseline.CONJUGATE_M),vectors[:,2:]])
    def builder(**kwargs):
        system = lrc.config_spaces(**kwargs)
        if index is not None:
            system['n'] = system['n'].copy()
            system['n'][12] = index
        return system
    shared,z,c,mag,image,objz = lrg.build_z_c_batch_ext(rows,FREE,builder=builder)
    tracer = lrg.GPULensTracerExt(shared,z,c)
    rms,alive = tracer.rms_batch_fast(mag,image,objz,baseline.FIELDS_SEARCH,ax,ay,use_ca=True)
    efl = baseline.paraxial_efl_batch(z,c,shared['nmed'])
    cpu = []
    for row in rows:
        system = builder(space1=row[0],space2=row[1],radii=dict(zip(FREE,row[3:])))
        cpu.append(lrc.paraxial_efl(system))
    error = np.max(np.abs(efl-np.array(cpu))/np.maximum(np.abs(cpu),1e-30))
    vig = 1-alive[:,-1]/np.maximum(alive[:,0],1)
    fno = baseline.fno_batch(tracer,shared['ca'][0],len(rows))
    objective = np.where(np.isfinite(rms).all(axis=1),(rms*baseline.FIELD_WEIGHTS).sum(axis=1),np.nan)
    return dict(rms=rms,n_alive=alive,efl=efl,vig=vig,fno=fno,obj=objective,mag=mag),float(error)


def hash_arrays(ev):
    digest = hashlib.sha256()
    for key in sorted(ev):
        array = np.ascontiguousarray(ev[key])
        digest.update(key.encode());digest.update(array.dtype.str.encode());digest.update(array.tobytes())
    return digest.hexdigest()


def measure():
    ax,ay = lrg.make_ray_grid(lrc.build(lrc.P)['ca'][0],n_ap=baseline.N_AP_OBJ,ca1_fill=1.05)
    reference = baseline.evaluate(baseline.PATENT_VECTOR[None,:],ax,ay)
    vectors = [PATENT.copy()];labels = ["nominal"]
    for idx in ADDED:
        for scale in (0.999,1.001):
            row = PATENT.copy();row[2+FREE.index(idx)] *= scale
            vectors.append(row);labels.append(f"radius{idx}*{scale}")
    ev,error = evaluate(np.array(vectors),ax,ay)
    nominal_exact = all(ev[k][:1].tobytes()==reference[k].tobytes() for k in reference)
    groups = [(labels,ev)]
    for index in INDEX_CHOICES:
        values,e = evaluate(PATENT[None,:],ax,ay,index=index)
        error = max(error,e);groups.append(([f"medium12={index}"],values))
    rows = [];hashes = []
    for labels,values in groups:
        hashes.append(hash_arrays(values))
        masks = baseline.certify(values,lrc.EFL_PATENT,float(reference['vig'][0]),float(reference['fno'][0]))
        for i,label in enumerate(labels):
            def number(key):
                x = float(values[key][i]);return x if np.isfinite(x) else None
            rows.append({"case":label,"objective_mm":number('obj'),"efl_mm":number('efl'),
                         "vignetting":number('vig'),"f_number":number('fno'),
                         "certified":bool(masks[0][i]),"efl_pass":bool(masks[1][i]),
                         "vignetting_pass":bool(masks[2][i]),"f_number_pass":bool(masks[3][i])})
    return {"rows":rows,"whole_array_hashes":hashes,"nominal_exact":nominal_exact,
            "max_cpu_paraxial_relative_error":error,"certified_count":sum(r['certified'] for r in rows)}


def main():
    a,b = measure(),measure()
    encoded = json.dumps(a,sort_keys=True,allow_nan=False).encode()
    gates = {"two_whole_results_identical":encoded==json.dumps(b,sort_keys=True,allow_nan=False).encode(),
             "nominal_matches_frozen":a['nominal_exact'],
             "all_cpu_paraxial_agree":a['max_cpu_paraxial_relative_error']<1e-12,
             "nominal_certified":a['rows'][0]['certified']}
    report = {"measurement":a,"sha256":hashlib.sha256(encoded).hexdigest(),"gates":gates,
              "scope":"single-index radius/glass sensitivity; no dispersion or manufacturing certificate"}
    (ROOT/'reports/lens_expansion_mechanism_probe.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == '__main__':
    raise SystemExit(main())
