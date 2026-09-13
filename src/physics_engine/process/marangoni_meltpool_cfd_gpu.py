"""Torch GPU momentum time loop for the unchanged Marangoni CFD model.

Conduction and energy sparse solves remain CPU operations once per outer step.
Momentum fields and pressure projection stay on GPU through the inner time loop.
Pressure uses a separable Neumann eigensolve with the ORIGINAL pinned first row,
not a different gauge equation or a dropped compatibility correction.
"""
import math
import types
import numpy as np
import torch
from physics_engine.process import marangoni_meltpool_cfd as cpu
from physics_engine.process.marangoni_meltpool_cfd import (
    TMELT, DTM, A_MUSH, EPS_PHI, MU, RHO, NU, THETA_M)


def zeros(shape):
    # Callers are in the momentum kernel namespace, bound per invocation.
    raise RuntimeError('device-specific allocator not bound')


def liquid_fraction(T):
    return torch.clamp((T-(TMELT-DTM))/(2*DTM),0.,1.)


class Pressure:
    def __init__(self,nx,nz,dx,dz,device='cuda'):
        self.shape=(nx,nz);self.device=device
        def basis(n):
            x=torch.arange(n,dtype=torch.float64,device=device)
            q=math.sqrt(2/n)*torch.cos(math.pi/n*(x[:,None]+.5)*x[None,:])
            q[:,0]=1/math.sqrt(n)
            return q
        self.qx,self.qz=basis(nx),basis(nz)
        ix=torch.arange(nx,dtype=torch.float64,device=device)
        iz=torch.arange(nz,dtype=torch.float64,device=device)
        lx=-4/dx**2*torch.sin(math.pi*ix/(2*nx))**2
        lz=-4/dz**2*torch.sin(math.pi*iz/(2*nz))**2
        self.eigenvalues=lx[:,None]+lz[None,:];self.eigenvalues[0,0]=1.

    def solve(self,rhs):
        b=torch.as_tensor(rhs,dtype=torch.float64,device=self.device).reshape(self.shape).clone()
        pin=b[0,0].clone();b[0,0].zero_();b[0,0]=-b.sum()
        modes=(self.qx.T@b@self.qz)/self.eigenvalues;modes[0,0].zero_()
        phi=self.qx@modes@self.qz.T
        return (phi-phi[0,0]+pin).reshape(-1)


def relax_momentum(u, w, T, nx, nz, dx, dz, dsdT, lu_p, max_sub=2000, vtol=1e-3):
    if dsdT == 0.0: return u, w, 0
    fL = liquid_fraction(T)
    fLu = 0.5 * (fL[0:nx - 1, :] + fL[1:nx, :]); Du = A_MUSH * (1 - fLu)**2 / (fLu**3 + EPS_PHI)
    fLw = 0.5 * (fL[:, 0:nz - 1] + fL[:, 1:nz]); Dw = A_MUSH * (1 - fLw)**2 / (fLw**3 + EPS_PHI)
    tau = -dsdT * (T[1:nx, 0] - T[0:nx - 1, 0]) / dx
    for sub in range(max_sub):
        umax = torch.abs(u).max().clamp_min(1e-9); wmax = torch.abs(w).max().clamp_min(1e-9)
        dt = (0.30 * torch.minimum(dx / umax, dz / wmax)).clamp_max(0.22 * min(dx, dz)**2 / NU); u_old_max = umax
        d2x_u = (u[2:nx + 1] - 2 * u[1:nx] + u[0:nx - 1]) / dx**2; u_int = u[1:nx, :]
        edge = zeros((nx - 1, nz + 1)); edge[:, 0] = tau
        edge[:, 1:nz] = MU * (u_int[:, 1:nz] - u_int[:, 0:nz - 1]) / dz; edge[:, nz] = MU * (-2 * u_int[:, nz - 1]) / dz
        visc_u = (MU * d2x_u + (edge[:, 1:] - edge[:, :-1]) / dz) / RHO
        uc = 0.5 * (u[:-1, :] + u[1:, :]); u_t = (1 - THETA_M) * uc + THETA_M * torch.where(uc > 0, u[:-1, :], u[1:, :])
        Fx = uc * u_t; dudt_x = -(Fx[1:nx, :] - Fx[0:nx - 1, :]) / dx
        we = 0.5 * (w[:-1, :] + w[1:, :]); We = we[:, 1:nz]
        u_cz = 0.5 * (u_int[:, :-1] + u_int[:, 1:]); u_tz = (1 - THETA_M) * u_cz + THETA_M * torch.where(We > 0, u_int[:, :-1], u_int[:, 1:])
        Fz = zeros((nx - 1, nz + 1)); Fz[:, 1:nz] = We * u_tz; dudt_z = -(Fz[:, 1:] - Fz[:, :-1]) / dz
        u_prov = (u_int + dt * (dudt_x + dudt_z + visc_u)) / (1 + dt * Du / RHO)
        w_int = w[:, 1:nz]; d2z_w = (w[:, 2:nz + 1] - 2 * w[:, 1:nz] + w[:, 0:nz - 1]) / dz**2
        wpad = torch.cat([w[0:1, :], w, -w[nx - 1:nx, :]], dim=0)
        d2x_w = (wpad[2:, 1:nz] - 2 * wpad[1:-1, 1:nz] + wpad[0:-2, 1:nz]) / dx**2
        visc_w = MU * (d2x_w + d2z_w) / RHO
        wc = 0.5 * (w[:, :-1] + w[:, 1:]); w_t = (1 - THETA_M) * wc + THETA_M * torch.where(wc > 0, w[:, :-1], w[:, 1:])
        Fzc = wc * w_t; dwdt_z = -(Fzc[:, 1:nz] - Fzc[:, 0:nz - 1]) / dz
        u_fz = 0.5 * (u[:, :-1] + u[:, 1:]); ue_int = u_fz[1:nx, :]
        wL, wR = w_int[0:nx - 1, :], w_int[1:nx, :]
        w_tx = (1 - THETA_M) * 0.5 * (wL + wR) + THETA_M * torch.where(ue_int > 0, wL, wR)
        Fxw = zeros((nx + 1, nz - 1)); Fxw[1:nx, :] = ue_int * w_tx; dwdt_x = -(Fxw[1:, :] - Fxw[:-1, :]) / dx
        w_prov = (w_int + dt * (dwdt_z + dwdt_x + visc_w)) / (1 + dt * Dw / RHO)
        ustar = u.clone(); ustar[1:nx, :] = u_prov; ustar[0, :] = 0.0; ustar[nx, :] = 0.0
        wstar = w.clone(); wstar[:, 1:nz] = w_prov; wstar[:, 0] = 0.0; wstar[:, nz] = 0.0
        div = (ustar[1:, :] - ustar[:-1, :]) / dx + (wstar[:, 1:] - wstar[:, :-1]) / dz
        rhs = (RHO / dt) * div.reshape(-1); rhs[0].zero_()
        phi = lu_p.solve(rhs).reshape(nx, nz); u = ustar; w = wstar
        u[1:nx, :] -= dt / RHO * (phi[1:, :] - phi[:-1, :]) / dx
        w[:, 1:nz] -= dt / RHO * (phi[:, 1:] - phi[:, :-1]) / dz
        stop = ~torch.all(torch.isfinite(u))
        if sub > 50:
            stop = stop | (torch.abs(torch.abs(u).max() - u_old_max) / u_old_max < vtol)
        if bool(stop): break
    return u, w, sub + 1

def momentum_step(u,w,T,nx,nz,dx,dz,dsdT,lu_p):
    fL = liquid_fraction(T)
    fLu = 0.5 * (fL[0:nx - 1, :] + fL[1:nx, :]); Du = A_MUSH * (1 - fLu)**2 / (fLu**3 + EPS_PHI)
    fLw = 0.5 * (fL[:, 0:nz - 1] + fL[:, 1:nz]); Dw = A_MUSH * (1 - fLw)**2 / (fLw**3 + EPS_PHI)
    tau = -dsdT * (T[1:nx, 0] - T[0:nx - 1, 0]) / dx
    umax = torch.abs(u).max().clamp_min(1e-9); wmax = torch.abs(w).max().clamp_min(1e-9)
    dt = (0.30 * torch.minimum(dx / umax, dz / wmax)).clamp_max(0.22 * min(dx, dz)**2 / NU); u_old_max = umax
    d2x_u = (u[2:nx + 1] - 2 * u[1:nx] + u[0:nx - 1]) / dx**2; u_int = u[1:nx, :]
    edge = zeros((nx - 1, nz + 1)); edge[:, 0] = tau
    edge[:, 1:nz] = MU * (u_int[:, 1:nz] - u_int[:, 0:nz - 1]) / dz; edge[:, nz] = MU * (-2 * u_int[:, nz - 1]) / dz
    visc_u = (MU * d2x_u + (edge[:, 1:] - edge[:, :-1]) / dz) / RHO
    uc = 0.5 * (u[:-1, :] + u[1:, :]); u_t = (1 - THETA_M) * uc + THETA_M * torch.where(uc > 0, u[:-1, :], u[1:, :])
    Fx = uc * u_t; dudt_x = -(Fx[1:nx, :] - Fx[0:nx - 1, :]) / dx
    we = 0.5 * (w[:-1, :] + w[1:, :]); We = we[:, 1:nz]
    u_cz = 0.5 * (u_int[:, :-1] + u_int[:, 1:]); u_tz = (1 - THETA_M) * u_cz + THETA_M * torch.where(We > 0, u_int[:, :-1], u_int[:, 1:])
    Fz = zeros((nx - 1, nz + 1)); Fz[:, 1:nz] = We * u_tz; dudt_z = -(Fz[:, 1:] - Fz[:, :-1]) / dz
    u_prov = (u_int + dt * (dudt_x + dudt_z + visc_u)) / (1 + dt * Du / RHO)
    w_int = w[:, 1:nz]; d2z_w = (w[:, 2:nz + 1] - 2 * w[:, 1:nz] + w[:, 0:nz - 1]) / dz**2
    wpad = torch.cat([w[0:1, :], w, -w[nx - 1:nx, :]], dim=0)
    d2x_w = (wpad[2:, 1:nz] - 2 * wpad[1:-1, 1:nz] + wpad[0:-2, 1:nz]) / dx**2
    visc_w = MU * (d2x_w + d2z_w) / RHO
    wc = 0.5 * (w[:, :-1] + w[:, 1:]); w_t = (1 - THETA_M) * wc + THETA_M * torch.where(wc > 0, w[:, :-1], w[:, 1:])
    Fzc = wc * w_t; dwdt_z = -(Fzc[:, 1:nz] - Fzc[:, 0:nz - 1]) / dz
    u_fz = 0.5 * (u[:, :-1] + u[:, 1:]); ue_int = u_fz[1:nx, :]
    wL, wR = w_int[0:nx - 1, :], w_int[1:nx, :]
    w_tx = (1 - THETA_M) * 0.5 * (wL + wR) + THETA_M * torch.where(ue_int > 0, wL, wR)
    Fxw = zeros((nx + 1, nz - 1)); Fxw[1:nx, :] = ue_int * w_tx; dwdt_x = -(Fxw[1:, :] - Fxw[:-1, :]) / dx
    w_prov = (w_int + dt * (dwdt_z + dwdt_x + visc_w)) / (1 + dt * Dw / RHO)
    ustar = u.clone(); ustar[1:nx, :] = u_prov; ustar[0, :] = 0.0; ustar[nx, :] = 0.0
    wstar = w.clone(); wstar[:, 1:nz] = w_prov; wstar[:, 0] = 0.0; wstar[:, nz] = 0.0
    div = (ustar[1:, :] - ustar[:-1, :]) / dx + (wstar[:, 1:] - wstar[:, :-1]) / dz
    rhs = (RHO / dt) * div.reshape(-1); rhs[0].zero_()
    phi = lu_p.solve(rhs).reshape(nx, nz); u = ustar; w = wstar
    u[1:nx, :] -= dt / RHO * (phi[1:, :] - phi[:-1, :]) / dx
    w[:, 1:nz] -= dt / RHO * (phi[:, 1:] - phi[:, :-1]) / dz
    relative=torch.abs(torch.abs(u).max()-u_old_max)/u_old_max
    invalid=~torch.all(torch.isfinite(u))
    return u,w,relative,invalid


class MomentumGraph:
    """Replay the unchanged tensor step with a fixed device buffer set."""
    def __init__(self,step,u,w,T,nx,nz,dx,dz,dsdT,pressure):
        self.u=u.clone();self.w=w.clone();self.T=T.clone()
        args=(self.u,self.w,self.T,nx,nz,dx,dz,dsdT,pressure)
        device=u.device
        stream=torch.cuda.Stream(device=device)
        stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(stream):
            for _ in range(2):step(*args)
        torch.cuda.current_stream(device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph,stream=stream):
            next_u,next_w,relative,invalid=step(*args)
            self.u.copy_(next_u);self.w.copy_(next_w)
            self.status=torch.stack((relative,invalid.to(torch.float64)))

    def run(self,u,w,T,max_sub,vtol):
        self.u.copy_(u);self.w.copy_(w);self.T.copy_(T)
        for sub in range(max_sub):
            self.graph.replay()
            status=self.status.cpu().numpy()
            if status[1] or (sub>50 and status[0]<vtol):break
        return self.u,self.w,sub+1


class Momentum:
    def __init__(self,device='cuda'):
        self.device=device;self.pressures={};self.graphs={}
        scope=dict(globals())
        scope['zeros']=lambda shape:torch.zeros(shape,dtype=torch.float64,device=device)
        self.run=types.FunctionType(relax_momentum.__code__,scope,relax_momentum.__name__,relax_momentum.__defaults__)
        self.step=types.FunctionType(momentum_step.__code__,scope,momentum_step.__name__)

    def __call__(self,u,w,T,nx,nz,dx,dz,dsdT,lu_p,max_sub=2000,vtol=1e-3):
        if dsdT==0.:return u,w,0
        key=(nx,nz,dx,dz)
        if key not in self.pressures:self.pressures[key]=Pressure(nx,nz,dx,dz,self.device)
        values=[torch.as_tensor(v,dtype=torch.float64,device=self.device) for v in (u,w,T)]
        if values[0].is_cuda:
            graph_key=(*key,dsdT)
            if graph_key not in self.graphs:
                self.graphs[graph_key]=MomentumGraph(self.step,*values,nx,nz,dx,dz,dsdT,self.pressures[key])
            u,w,n=self.graphs[graph_key].run(*values,max_sub,vtol)
        else:
            u,w,n=self.run(*values,nx,nz,dx,dz,dsdT,self.pressures[key],max_sub,vtol)
        return u.detach().cpu().numpy(),w.detach().cpu().numpy(),n


def tune_q0(nx,nz,dx,dz,target_hw=75e-6):
    """Reuse the fixed conduction matrix within the original 40-step calibration."""
    xc=(np.arange(nx)+0.5)*dx
    matrix,boundary=cpu.build_conduction(nx,nz,dx,dz,0.)
    factor=cpu.spla.splu(matrix.tocsc())
    source_shape=np.exp(-2*xc**2/cpu.A_BEAM**2)
    lo,hi=1e7,5e9
    for _ in range(40):
        mid=0.5*(lo+hi)
        rhs=boundary.copy()
        # Match the source multiplication and boundary-addition order exactly.
        rhs[::nz]+=(mid*source_shape)*dx
        T=factor.solve(rhs).reshape(nx,nz)
        if cpu.half_width(T[:,0],xc)<target_hw:lo=mid
        else:hi=mid
    return 0.5*(lo+hi)


def solver_scope(device='cuda'):
    scope=dict(vars(cpu));scope['relax_momentum']=Momentum(device)
    # Momentum owns its separable pressure operator; the CPU LU would be unused.
    scope['build_pressure']=lambda *args:None
    scope['tune_q0']=tune_q0
    scope['segregated_solve']=types.FunctionType(cpu.segregated_solve.__code__,scope,
        cpu.segregated_solve.__name__,cpu.segregated_solve.__defaults__)
    scope['width_ratio']=types.FunctionType(cpu.width_ratio.__code__,scope,
        cpu.width_ratio.__name__,cpu.width_ratio.__defaults__)
    return scope


def width_ratio(dsdt,nx=100,nz=55,want=('ratio',),device='cuda'):
    return solver_scope(device)['width_ratio'](dsdt,nx,nz,want)


def main(device='cuda'):
    return types.FunctionType(cpu.main.__code__,solver_scope(device),cpu.main.__name__)()

if __name__=='__main__':raise SystemExit(main())
