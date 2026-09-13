"""Original ILT knob search using the existing batched SOCS GPU apply.

All 35 original mask candidates and nine original focus values are evaluated in
batches. Every source-span SOCS kernel is retained (no 8-kernel approximation).
Threshold selection, edge matching, tie order, gates, and Jacobian/SVD diagnostics
remain the CPU sibling's functions. This is a hybrid ILT loop, not a resident
optimizer. CPU kernel setup and transfers belong in end-to-end timings.
"""
import types
import numpy as np
import torch
from physics_engine.litho import d_litho_certified_imaging_ilt as cpu
from physics_engine.litho.socs_batched_tcc_apply_gpu_api_for_ilt import batched_aerial


def kernels(n, dx, sigma, defocus=0., source=None, k0=cpu.K0):
    offsets, weights = source if source is not None else cpu.source_line(n, dx, sigma, k0)
    weights = np.asarray(weights, dtype=float)
    if not len(offsets) or len(offsets) != len(weights) or not np.isfinite(weights).all() or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('nonempty source with finite nonnegative weights required')
    pupil = cpu.pupil_1d(n, dx, k0, defocus)
    columns = np.stack([np.roll(pupil, -int(b)) for b in offsets], axis=1)
    columns *= np.sqrt(weights / weights.sum())[None, :]
    # TCC = columns @ columns.H. Thin SVD retains the full source span.
    u, s, _ = np.linalg.svd(columns, full_matrices=False)
    return u.T, s ** 2


class Imaging:
    def __init__(self, device='cuda'):
        self.device = device
        self.kernel_cache = {}
        self.images = {}
        self.candidate_images = None
        self.last_jacobian = None

    def batch(self, masks, dx, sigma, defocus=0., source=None, k0=cpu.K0, tile=0):
        masks = np.asarray(masks, dtype=float)
        if masks.ndim != 2 or not len(masks) or not np.isfinite(masks).all():
            raise ValueError('finite nonempty (batch, pixels) masks required')
        source = source if source is not None else cpu.source_line(masks.shape[1], dx, sigma, k0)
        key = (masks.shape[1], dx, sigma, float(defocus), k0, tuple(source[0]), tuple(source[1]))
        if key not in self.kernel_cache:
            self.kernel_cache[key] = kernels(masks.shape[1], dx, sigma, defocus, source, k0)
        phi, lam = self.kernel_cache[key]
        return batched_aerial(masks, phi, lam, self.device, torch.float64, tile=tile)

    def aerial(self, mask, dx, sigma, k0=cpu.K0, defocus=0., source=None, dose=1.):
        mask=np.asarray(mask,dtype=float)
        source = source if source is not None else cpu.source_line(len(mask), dx, sigma, k0)
        key=(mask.tobytes(),dx,sigma,k0,float(defocus),tuple(source[0]),tuple(source[1]))
        if key not in self.images:
            self.images[key]=self.batch(mask[None],dx,sigma,defocus,source,k0)[0]
        return dose*self.images[key]

    def jacobian(self,mask0,dx,sigma,k0=cpu.K0,src=None,eps=1e-3):
        n=len(mask0)
        if src is None:src=cpu.source_line(n,dx,sigma,k0)
        masks=np.repeat(np.asarray(mask0)[None,:],2*n,axis=0)
        for j in range(n):
            masks[2*j,j]+=eps
            masks[2*j+1,j]-=eps
        # The CPU finite-difference function uses default K0 for aerial application,
        # even when k0 is supplied for source construction; retain that convention.
        images=self.batch(masks,dx,sigma,source=src,tile=64)
        self.last_jacobian=((images[0::2]-images[1::2])/(2*eps)).T.copy()
        return self.last_jacobian

    def process_window(self,mask,dx,sigma,thr,ideal_edges,ctr_win,tol,src,
                       doses=None,focuses=None):
        """Evaluate all dose/focus crossing distances with the original inequalities."""
        if doses is None:doses=np.linspace(.94,1.06,7)
        if focuses is None:focuses=np.linspace(-60.,60.,7)
        count=len(doses)*len(focuses)
        if count==0:raise ZeroDivisionError('division by zero')
        base=np.array([self.aerial(mask,dx,sigma,defocus=z,source=src) for z in focuses])
        images=(np.asarray(doses)[None,:,None]*base[:,None,:]).reshape(count,len(mask))
        a=images[:,:-1]-thr;b=images[:,1:]-thr
        zero=a==0.;cross=(~zero)&(a*b<0.)
        indices=np.broadcast_to(np.arange(len(mask)-1),a.shape)
        positions=np.full(a.shape,np.inf)
        positions[zero]=indices[zero]*dx
        fraction=a[cross]/(a[cross]-b[cross])
        positions[cross]=(indices[cross]+fraction)*dx
        inside=(ctr_win[0]<positions)&(positions<ctr_win[1])
        has_edges=inside.any(axis=1)
        good=np.ones(count,dtype=bool)
        for target in ideal_edges:
            distance=np.where(inside,np.abs(positions-target),np.inf)
            nearest=distance.min(axis=1,initial=np.inf)
            good &= has_edges & ~(nearest>tol)
        return int(good.sum())/count

    def prepare_candidates(self):
        configs=[(None,0.)]+[(o,w) for o in (70.,90.,110.) for w in (16.,22.)]
        masks=np.array([cpu._mask_from_knobs(256,2.,128,45.,b,o,w)
                        for b in (-6.,-3.,0.,3.,6.) for o,w in configs])
        source=cpu.source_line(256,2.,.6)
        cube=[]
        for focus in np.linspace(-90.,90.,9):
            batch=self.batch(masks,2.,.6,focus,source)
            cube.append(batch)
            for mask,img in zip(masks,batch):
                self.images[(mask.tobytes(),2.,.6,cpu.K0,float(focus),tuple(source[0]),tuple(source[1]))]=img
        self.candidate_images=np.array(cube)
        return masks


def find_edges_1d(img,dx,thr):
    """Select crossing intervals in parallel; preserve scalar interpolation order."""
    if len(img)<2:return []
    img=np.asarray(img)
    # Scalar promotion differs between NumPy versions; match the reference scalar.
    dtype=np.asarray(img[0]-thr).dtype
    a=img[:-1].astype(dtype,copy=False)-thr
    b=img[1:].astype(dtype,copy=False)-thr
    indices=np.flatnonzero((a==0.) | (a*b<0.))
    edges=[]
    for index in indices:
        i=int(index)
        if a[i]==0.:
            edges.append((i*dx,np.sign(img[i+1]-img[i])))
        else:
            frac=a[i]/(a[i]-b[i])
            edges.append(((i+frac)*dx,np.sign(b[i]-a[i])))
    return edges


def section_c(device='cuda',return_jacobian=False):
    engine=Imaging(device)
    masks=engine.prepare_candidates()
    # Isolated function globals avoid modifying the CPU module or its gates.
    scope=dict(vars(cpu)); scope['EV']={}; scope['aerial_1d']=engine.aerial
    scope['find_edges_1d']=find_edges_1d
    scope['jacobian_1d']=engine.jacobian
    scope['process_window_1d']=engine.process_window
    run=types.FunctionType(cpu.section_c.__code__,scope,cpu.section_c.__name__)
    result=run()
    if return_jacobian:return result,engine.candidate_images,masks,engine.last_jacobian
    return result, engine.candidate_images, masks
