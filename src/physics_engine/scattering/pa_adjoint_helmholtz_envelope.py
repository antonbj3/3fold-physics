"""
Helmholtz adjoint z (L^T z = g, point g): separate the RADIAL ENVELOPE of |z| from the
oscillation, and count sign changes along a ray to confirm oscillatory (non-decaying) vs
decaying behavior. Compare diffusion (control) vs Helmholtz at several kappa.

Diagnostics per case:
  - angular mean of |z| on radius shells (envelope), and its decay fit
  - number of sign changes of z(r) along +x ray (oscillation count)
  - measured oscillation wavelength vs theoretical 2pi/kappa
Reuses the same FD operator (5-point Laplacian, Dirichlet zero BC).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

def idx(i, j, n): return i * n + j

def build_L(n, h, kappa=0.0):
    N = n * n
    rows=[]; cols=[]; vals=[]
    diff = 1.0/(h*h)
    for i in range(n):
        for j in range(n):
            k = idx(i,j,n)
            if i in (0,n-1) or j in (0,n-1):
                rows.append(k); cols.append(k); vals.append(1.0); continue
            diag = 4.0*diff - kappa*kappa
            for (ii,jj) in [(i+1,j),(i-1,j),(i,j+1),(i,j-1)]:
                rows.append(k); cols.append(idx(ii,jj,n)); vals.append(-diff)
            rows.append(k); cols.append(k); vals.append(diag)
    return sp.csr_matrix((vals,(rows,cols)), shape=(N,N))

def solve_adj(L, n, src):
    N=n*n; g=np.zeros(N); g[idx(*src,n)]=1.0
    return spla.spsolve(L.T.tocsc(), g).reshape(n,n)

def shell_envelope(z, n, h, src, radii, nsamp=72):
    ci,cj=src; env=[]
    for r in radii:
        acc=[]
        for t in np.linspace(0,2*np.pi,nsamp,endpoint=False):
            fi=ci+(r/h)*np.cos(t); fj=cj+(r/h)*np.sin(t)
            i0=int(np.floor(fi)); j0=int(np.floor(fj))
            if i0<0 or j0<0 or i0+1>=n or j0+1>=n: continue
            ti=fi-i0; tj=fj-j0
            v=((1-ti)*(1-tj)*z[i0,j0]+ti*(1-tj)*z[i0+1,j0]
               +(1-ti)*tj*z[i0,j0+1]+ti*tj*z[i0+1,j0+1])
            acc.append(abs(v))
        env.append(np.mean(acc) if acc else np.nan)
    return np.array(env)

def ray_signed(z, n, h, src, angle, rmax, nr=200):
    ci,cj=src; a=np.deg2rad(angle); di,dj=np.cos(a),np.sin(a)
    rs=np.linspace(0.002,rmax,nr); vs=[]
    for r in rs:
        fi=ci+(r/h)*di; fj=cj+(r/h)*dj
        i0=int(np.floor(fi)); j0=int(np.floor(fj))
        if i0<0 or j0<0 or i0+1>=n or j0+1>=n: vs.append(np.nan); continue
        ti=fi-i0; tj=fj-j0
        v=((1-ti)*(1-tj)*z[i0,j0]+ti*(1-tj)*z[i0+1,j0]
           +(1-ti)*tj*z[i0,j0+1]+ti*tj*z[i0+1,j0+1])
        vs.append(v)
    return rs, np.array(vs)

def sign_changes(rs, vs):
    m=np.isfinite(vs); r=rs[m]; v=vs[m]
    s=np.sign(v); chg=np.where(np.diff(s)!=0)[0]
    # zero-crossing radii (linear interp)
    zr=[]
    for k in chg:
        if v[k]!=v[k+1]:
            f=v[k]/(v[k]-v[k+1]); zr.append(r[k]+f*(r[k+1]-r[k]))
    return len(chg), np.array(zr)

def env_decay_p(radii, env):
    m=np.isfinite(env)&(env>0); r=radii[m]; e=env[m]
    if len(r)<3: return np.nan
    A=np.vstack([np.ones_like(r),-np.log(r)]).T
    c,*_=np.linalg.lstsq(A,np.log(e),rcond=None)
    return c[1]

def main():
    n=201; h=1.0/(n-1); src=(n//2,n//2)
    radii=np.array([0.04,0.06,0.08,0.10,0.13,0.16,0.20,0.24,0.28,0.32,0.36,0.40])
    print("Envelope = angular mean of |z| on radius shell. p>0 => decaying envelope.\n")
    for label,kappa in [("DIFFUSION(ctrl)",0.0),("HELM k=10",10.0),("HELM k=20",20.0),
                        ("HELM k=40",40.0),("HELM k=70",70.0)]:
        L=build_L(n,h,kappa); z=solve_adj(L,n,src)
        env=shell_envelope(z,n,h,src,radii)
        p=env_decay_p(radii,env)
        rs,vs=ray_signed(z,n,h,src,0.0,0.45)
        nsc,zr=sign_changes(rs,vs)
        if len(zr)>=2:
            meas_halfwave=np.mean(np.diff(zr))  # spacing between zeros = half wavelength
            meas_lam=2*meas_halfwave
        else:
            meas_lam=np.nan
        theo_lam=2*np.pi/kappa if kappa>0 else np.inf
        print(f"{label:16s} kappa={kappa:5.1f}  env_decay_p={p:6.3f}  "
              f"sign_changes(+x ray to r=0.45)={nsc:2d}  "
              f"meas_lambda={meas_lam:6.4f} theo_lambda={theo_lam:6.4f}")
        envstr=" ".join(f"{e:.2e}" for e in env)
        print(f"     shells r={list(radii)}")
        print(f"     env(|z|): {envstr}")
    print("\nClassification: kappa=0 -> monotone-decaying envelope, 0 sign changes (frustum: smooth bump).")
    print("kappa>0 -> non-monotone / sign changes => OSCILLATORY (propagating), envelope ~ near-flat (2D Green ~ r^-1/2).")

if __name__=="__main__":
    main()
