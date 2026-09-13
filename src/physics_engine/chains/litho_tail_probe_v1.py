"""Three/four-stage uncertainty and rare-pore falsifier; synthetic S4 only."""
import hashlib
import json
from pathlib import Path
import numpy as np
import litho_chain_v1 as base


def digest(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def pore(widths, uniforms, radii, scale=1., probability=.02):
    return np.maximum(0., widths-np.where(uniforms<probability,2.*radii*scale,0.))


def tail(a):
    return float(np.quantile(np.abs(a-np.median(a)),.99))


def etched_lines(L,W,scale):
    lateral=[]
    for w in W:
        _,h,_=base.etch_levelset(w,v0=base.V0_ETCH*scale,dt=base.ETCH_DT,nsteps=base.ETCH_NSTEPS)
        ar=h/w
        vel=base.V0_ETCH*scale*base.phi_neutral(ar)*(1.-base.phi_ion(np.maximum(ar,1e-6),base.SIGMA_ION))
        lateral.append(float(np.sum(vel)*base.ETCH_DT))
    return L-2.*np.asarray(lateral)


def observe():
    rng=np.random.default_rng(20260913)
    perturb=rng.uniform(-1.,1.,(64,6))
    arms={}
    parity=0.
    for arm in ('S1','S2','S3','S4','all'):
        means3=[]; means4=[]; widths3=[]; widths4=[]; hashes=[]
        for draw,z in enumerate(perturb):
            na,so,si,dose,blur,es,ps=base.NA0,base.SIGMA_OUT0,base.SIGMA_IN0,base.DOSE0,base.BLUR0,1.,1.
            if arm in ('S1','all'):
                na*=1.+base.TOL['na']*z[0];so*=1.+base.TOL['sigma']*z[1];si*=1.+base.TOL['sigma']*z[1]
            if arm in ('S2','all'):
                dose*=1.+base.TOL['dose']*z[2];blur*=1.+base.TOL['blur']*z[3]
            if arm in ('S3','all'):es*=1.+base.TOL['etch']*z[4]
            if arm in ('S4','all'):ps*=1.+.2*z[5]
            _,mask=base.make_mask()
            image,_=base.aerial(mask,na,so,si)
            L,W,_,_=base.resist_lines(image,dose,blur,np.random.default_rng(1000+draw))
            three=etched_lines(L,W,es)
            if draw==0:
                expected=base.etch_feature(L,W,v0=base.V0_ETCH*es)
                parity=max(parity,abs(float(three.mean())-expected[0]),abs(float(3.*three.std()/np.sqrt(2.))-expected[1]))
            prng=np.random.default_rng(9000+draw)
            uniforms=prng.random(len(three));radii=prng.lognormal(np.log(2.),.8,len(three))
            four=pore(three,uniforms,radii,ps)
            means3.append([float(three.mean()),float(3.*three.std()/np.sqrt(2.))])
            means4.append([float(four.mean()),float(3.*four.std()/np.sqrt(2.))])
            widths3.extend(three);widths4.extend(four)
            hashes.append([digest(x) for x in (image,L,W,three,uniforms,radii,four)])
        arms[arm]=dict(three=means3,four=means4,p99_three=tail(np.asarray(widths3)),p99_four=tail(np.asarray(widths4)),hashes=hashes)
    ratios={}
    for mode,stages in (('three',('S1','S2','S3')),('four',('S1','S2','S3','S4'))):
        isolated=np.array([np.std(arms[s][mode],axis=0) for s in stages])
        joint=np.std(arms['all'][mode],axis=0)
        ratios[mode]=dict(joint=joint.tolist(),linear_sum=isolated.sum(axis=0).tolist(),
            ratio=(joint/isolated.sum(axis=0)).tolist(),
            dominant=[stages[i] for i in isolated.argmax(axis=0)],
            sublinear=(joint<=isolated.sum(axis=0)+1e-12).tolist())
    return dict(arms=arms,composition=ratios,aggregate_parity_error=parity,
        pore_tail_rewidens=arms['all']['p99_four']>arms['all']['p99_three'])


def report():
    a,b=observe(),observe()
    nom=base.run_chain(seed=7);double=base.run_chain(dose=2*base.DOSE0,seed=7)
    g1,epe,rel,device=base.stage_gate_s1()
    g2,slope,cv=base.stage_gate_s2(nom['ils'])
    g3,lag=base.stage_gate_s3()
    control=np.full(10000,45.);u=np.ones(10000);u[:200]=0.;r=np.full(10000,5.)
    gates=dict(exact_repeat=a==b,aggregate_parity=a['aggregate_parity_error']<=1e-12,
        stage1=bool(g1),stage2=bool(g2),stage3=bool(g3),
        pore_null=np.array_equal(pore(control,u,r,probability=0.),control),
        tail_detector=tail(pore(control,u,r))>tail(control),
        clear_null=base.run_chain(clear_mask=True)['line'] is False,
        dose_cd=double['cd_etch']<nom['cd_etch'],dose_ler=double['ler_etch']<nom['ler_etch'],
        ratios_attributed=all(len(v['dominant'])==2 and np.isfinite(v['ratio']).all() for v in a['composition'].values()))
    gates={k:bool(v) for k,v in gates.items()}
    return dict(scope='numerical composition observer; synthetic uncalibrated pore stress test',observation=a,gates=gates,
        stage_evidence=dict(epe=epe,relative=rel,device=device,slope=slope,cv=cv,lag=lag),
        status='VERIFIED-FRESH' if all(gates.values()) else 'OWN-GATE-FAIL')


if __name__=='__main__':
    result=report()
    path=Path('reports/litho_tail_probe_v1.json');path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(gates=result['gates'],composition=result['observation']['composition'],
        tail={k:result['observation']['arms']['all'][k] for k in ('p99_three','p99_four')},status=result['status'])))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
