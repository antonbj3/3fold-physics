"""Six-coordinate continuation and separately certified fixed-index families."""
import hashlib
import json
from pathlib import Path

import numpy as np

import lens_expansion_mechanism_probe as mechanism

baseline = mechanism.baseline
ACTIVE = (9,13,8,11)
CENTER = np.array([baseline.PATENT_SPACE1,baseline.PATENT_SPACE2,
                   *[baseline.lrc.P[i][0] for i in ACTIVE]],np.float64)
extra = np.sort(np.array([[baseline.lrc.P[i][0]*0.997,baseline.lrc.P[i][0]*1.003] for i in (8,11)]),axis=1)
LOW = np.r_[baseline.LO,extra[:,0]]
HIGH = np.r_[baseline.HI,extra[:,1]]


def expanded(vectors):
    out = np.tile(mechanism.PATENT,(len(vectors),1))
    out[:,:2] = vectors[:,:2]
    for column,idx in enumerate(ACTIVE):
        out[:,2+mechanism.FREE.index(idx)] = vectors[:,2+column]
    return out


def plain(value):
    if isinstance(value,np.ndarray): return value.tolist()
    if isinstance(value,dict): return {k:plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)): return [plain(v) for v in value]
    if isinstance(value,np.generic): return value.item()
    return value


def search(ax,ay,refs,incumbent,*,index=None,size=4096,generations=6,collapsed=False):
    low = CENTER if collapsed else LOW
    high = CENTER if collapsed else HIGH
    rng = np.random.default_rng(baseline.SEARCH_SEED)
    pop = low+rng.random((size,6))*(high-low)
    pop[0] = CENTER
    if size>1: pop[1] = np.clip(incumbent,low,high)
    best = None;certified = rejected = 0;logs = [];digest = hashlib.sha256()
    for generation in range(generations):
        ev,error = mechanism.evaluate(expanded(pop),ax,ay,index=index)
        if error>=1e-12: raise RuntimeError('CPU paraxial instrument mismatch')
        masks = baseline.certify(ev,baseline.lrc.EFL_PATENT,refs[1],refs[2])
        ok = masks[0];certified += int(ok.sum());rejected += int((~ok).sum())
        digest.update(pop.tobytes());digest.update(mechanism.hash_arrays(ev).encode());digest.update(ok.tobytes())
        eligible = np.flatnonzero(ok)
        if len(eligible):
            order = eligible[np.argsort(ev['obj'][eligible],kind='stable')]
            lead = order[0]
            if best is None or float(ev['obj'][lead])<best['objective_mm']:
                best = {'objective_mm':float(ev['obj'][lead]),'vector':pop[lead].tolist(),
                        'efl_mm':float(ev['efl'][lead]),'vignetting':float(ev['vig'][lead]),
                        'f_number':float(ev['fno'][lead]),'rms_mm':ev['rms'][lead].tolist(),
                        'certified':bool(ok[lead])}
            elite = pop[order[:min(baseline.N_ELITE,len(order))]]
            width = np.maximum(elite.std(axis=0),baseline.SIGMA0*baseline.SIGMA_DECAY**generation*(high-low)*0.025)
            proposals = elite.mean(axis=0)+width*rng.standard_normal((size-len(elite),6))
            pop = np.clip(np.vstack([elite,proposals]),low,high)
        else:
            pop = low+rng.random((size,6))*(high-low)
        logs.append({'generation':generation,'certified':int(ok.sum()),
                     'best_objective_mm':None if best is None else best['objective_mm']})
    return {'index':index,'best':best,'certified':certified,'rejected':rejected,
            'evaluated':size*generations,'generation_sha256':digest.hexdigest(),'log':logs}


def measure():
    ax,ay = baseline.lrg.make_ray_grid(baseline.lrc.build(baseline.lrc.P)['ca'][0],n_ap=baseline.N_AP_OBJ,ca1_fill=1.05)
    refs = baseline.reference_values(ax,ay)
    original = baseline.search(refs=refs,verbose=False)
    if original['best_vec'] is None: raise RuntimeError('Frozen baseline has no certified winner')
    incumbent = np.r_[original['best_vec'],CENTER[4:]]
    continuation = search(ax,ay,refs,incumbent)
    families = [search(ax,ay,refs,incumbent,index=index,size=512,generations=4) for index in mechanism.INDEX_CHOICES]
    null = search(ax,ay,refs,CENTER,size=8,generations=2,collapsed=True)
    return {'frozen_baseline':plain(original),'continuation':continuation,'fixed_index_families':families,
            'collapsed_null':null,'patent_objective_mm':refs[3]}


def main():
    a,b = measure(),measure()
    encoded = json.dumps(a,sort_keys=True,allow_nan=False).encode()
    runs = [a['continuation'],*a['fixed_index_families'],a['collapsed_null']]
    best = a['continuation']['best']
    gates = {'two_full_sweeps_identical':encoded==json.dumps(b,sort_keys=True,allow_nan=False).encode(),
             'all_reported_winners_certified':all(r['best'] is None or r['best']['certified'] for r in runs),
             'strict_continuation_improvement':best is not None and best['objective_mm']<a['frozen_baseline']['best_obj'],
             'exact_collapsed_null':a['collapsed_null']['best']['objective_mm']==a['patent_objective_mm'],
             'all_candidates_accounted':all(r['certified']+r['rejected']==r['evaluated'] for r in runs)}
    report = {'measurement':a,'sha256':hashlib.sha256(encoded).hexdigest(),'gates':gates,
              'scope':'single-index continuation; extra search budget; no dispersion/manufacturing certificate'}
    (mechanism.ROOT/'reports/lens_design_search_expanded_v1.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'gates':gates,'frozen_best':a['frozen_baseline']['best_obj'],
                      'expanded_best':best,'family_counts':[(r['index'],r['certified'],r['rejected']) for r in a['fixed_index_families']]},indent=2))
    return 0 if all(gates.values()) else 1


if __name__=='__main__':
    raise SystemExit(main())
