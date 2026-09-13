"""Warp float64 PLIC/Weymouth-Yue sibling; original stencil and split order.

Two kernels per directional sweep compute donor-cell geometric face fluxes and
apply the original compression/clipping update. No atomics or reordered temporal
steps. Host fixture generation is the unchanged CPU supersampled disk.
"""
import numpy as np
import warp as wp
from physics_engine.litho import vof_zalesak_cert as cpu
wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})
D=wp.float64

@wp.func
def sample(F:wp.array2d(dtype=D),i:int,j:int):
    n=F.shape[0]
    return F[wp.clamp(i,0,n-1),wp.clamp(j,0,n-1)]

@wp.func
def normal(F:wp.array2d(dtype=D),i:int,j:int):
    fx=sample(F,i+1,j-1)+D(2)*sample(F,i+1,j)+sample(F,i+1,j+1)-sample(F,i-1,j-1)-D(2)*sample(F,i-1,j)-sample(F,i-1,j+1)
    fy=sample(F,i-1,j+1)+D(2)*sample(F,i,j+1)+sample(F,i+1,j+1)-sample(F,i-1,j-1)-D(2)*sample(F,i,j-1)-sample(F,i+1,j-1)
    nx=-fx;ny=-fy;s=wp.abs(nx)+wp.abs(ny)
    if s<D(1e-12):return wp.vec2d(D(1),D(0))
    return wp.vec2d(nx/s,ny/s)

@wp.func
def alpha(nx:D,ny:D,F:D):
    mx=wp.abs(nx);my=wp.abs(ny);s=mx+my
    mxn=mx/s;myn=my/s;m1=wp.min(mxn,myn);m2=wp.max(mxn,myn)
    V=wp.min(F,D(1)-F);Vc=D(0)
    if m2>D(1e-12):Vc=m1/(D(2)*m2)
    a=V*m2+m1/D(2)
    if V<=Vc:a=wp.sqrt(D(2)*m1*m2*V)
    if F>D(.5):a=D(1)-a
    return a*s+wp.min(nx,D(0))+wp.min(ny,D(0))

@wp.func
def volume(nx:D,ny:D,alpha:D):
    a=alpha-wp.min(nx,D(0))-wp.min(ny,D(0));mx=wp.abs(nx);my=wp.abs(ny)
    if a<=D(0):return D(0)
    if a>=mx+my:return D(1)
    if mx<D(1e-12):return wp.clamp(a/my,D(0),D(1))
    if my<D(1e-12):return wp.clamp(a/mx,D(0),D(1))
    p=wp.max(a-mx,D(0));q=wp.max(a-my,D(0))
    return wp.clamp((a*a-p*p-q*q)/(D(2)*mx*my),D(0),D(1))

@wp.kernel
def flux_kernel(F:wp.array2d(dtype=D),c:wp.array2d(dtype=D),axis:int,geom:int,flux:wp.array2d(dtype=D)):
    face,other=wp.tid();n=F.shape[0];value=D(0)
    if face>0 and face<n:
        cf=c[face,other];donor=face
        if cf>D(0):donor=face-1
        i=donor;j=other
        if axis==1:i=other;j=donor
        fraction=F[i,j]
        if geom==0:
            value=cf*fraction
        elif wp.abs(cf)>D(1e-14):
            value=cf*fraction
            if fraction>D(1e-6) and fraction<D(1)-D(1e-6):
                norm=normal(F,i,j);nx=norm[0];ny=norm[1]
                a=alpha(nx,ny,fraction)
                if axis==1:
                    tmp=nx;nx=ny;ny=tmp
                if cf>D(0):
                    x0=D(1)-cf
                    width=D(1)-x0
                    value=width*volume(nx*width,ny,a-nx*x0)
                else:
                    value=cf*volume(nx*(-cf),ny,a)
    flux[face,other]=value

@wp.kernel
def update_kernel(F:wp.array2d(dtype=D),c:wp.array2d(dtype=D),flux:wp.array2d(dtype=D),axis:int,geom:int,out:wp.array2d(dtype=D)):
    i,j=wp.tid();k=i;o=j
    if axis==1:k=j;o=i
    value=F[i,j]-(flux[k+1,o]-flux[k,o])
    if geom==1:value=value+F[i,j]*(c[k+1,o]-c[k,o])
    out[i,j]=wp.clamp(value,D(0),D(1))


def do_sweep(F,c,axis,scheme='geom',device='cuda'):
    F=np.asarray(F,dtype=np.float64);c=np.asarray(c,dtype=np.float64)
    if F.ndim!=2 or F.shape[0]!=F.shape[1] or axis not in (0,1):raise ValueError('square field and axis 0/1 required')
    n=len(F)
    if c.shape!=((n+1,n) if axis==0 else (n,n+1)) or not np.isfinite(c).all() or np.max(abs(c))>1:
        raise ValueError('finite face Courant numbers with magnitude <= 1 required')
    if not np.isfinite(F).all() or np.min(F)<0 or np.max(F)>1 or scheme not in ('geom','upwind'):
        raise ValueError('bounded finite fractions and supported scheme required')
    field=wp.array(F,dtype=D,device=device);courant=wp.array(c if axis==0 else c.T.copy(),dtype=D,device=device)
    flux=wp.empty((n+1,n),dtype=D,device=device);out=wp.empty((n,n),dtype=D,device=device)
    wp.launch(flux_kernel,(n+1,n),inputs=[field,courant,axis,int(scheme=='geom'),flux],device=device)
    wp.launch(update_kernel,(n,n),inputs=[field,courant,flux,axis,int(scheme=='geom'),out],device=device)
    return out.numpy()


def rotate(N,scheme='geom',revs=1.,cfl=.5,device='cuda',return_field=False):
    if not isinstance(N,int) or N<3 or not np.isfinite([revs,cfl]).all() or revs<=0 or not 0<cfl<=1 or scheme not in ('geom','upwind'):
        raise ValueError('valid grid, positive revolutions and 0 < CFL <= 1 required')
    dx=1./N;F0=cpu.zalesak_F(N);A0=F0.sum();om=2*np.pi
    yc=(np.arange(N)+.5)*dx
    dt0=cfl*dx/(om*.5*np.sqrt(2));steps=max(1,int(round(revs/dt0)));dt=revs/steps
    # Match CPU operation order in the original Courant construction.
    cx=np.tile((-om*(yc-.5))*dt/dx,(N+1,1))
    cy=np.tile((om*(yc-.5))*dt/dx,(N+1,1))
    field=wp.array(F0,dtype=D,device=device);out=wp.empty_like(field)
    courants=[wp.array(c,dtype=D,device=device) for c in (cx,cy)]
    flux=wp.empty((N+1,N),dtype=D,device=device)
    for step in range(steps):
        for axis in ((0,1) if step%2==0 else (1,0)):
            wp.launch(flux_kernel,(N+1,N),inputs=[field,courants[axis],axis,int(scheme=='geom'),flux],device=device)
            wp.launch(update_kernel,(N,N),inputs=[field,courants[axis],flux,axis,int(scheme=='geom'),out],device=device)
            field,out=out,field
    F=field.numpy()
    result=dict(area=float(F.sum()*dx*dx),consv=float(abs(F.sum()-A0)/A0),
        thick=int(((F>.01)&(F<.99)).sum())/max(int(((F0>.01)&(F0<.99)).sum()),1),
        l1=float(np.abs(F-F0).sum()/A0),nsteps=steps)
    return (result,F) if return_field else result
