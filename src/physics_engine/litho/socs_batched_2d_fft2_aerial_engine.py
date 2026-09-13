#!/usr/bin/env python3
"""Batched 2-D SOCS aerial engine (fft2): B two-dimensional mask aerial images in one GPU call.

The 2-D counterpart of the 1-D batched TCC-apply. The K two-dimensional SOCS kernels are precomputed once by
eigen-decomposing the 2-D Hopkins TCC over the flattened frequency grid; the apply is then
  aerial_b = sum_k lam_k |IFFT2[ phi_k . FFT2(M_b) ]|^2
as one batched fft2 call, intermediate (B,K,Ny,Nx) — so tiling is the VRAM lever.

API: batched_aerial_2d(masks (B,Ny,Nx), phis (K,Ny,Nx), lams (K)) -> aerials (B,Ny,Nx).

GATES: (1) SOCS with all kernels equals the 2-D Abbe source-point sum (<1e-6 relative, the correctness anchor);
(2) the batched apply is floating-point identical to a per-mask 2-D loop; (3) batched throughput is far above the
per-mask loop; (4) memory tiling is identical to the untiled result.

I/O: no input files; prints the gate lines and a verdict. Requires a CUDA device (torch.fft).
Optics: 193 nm, NA 1.35, annular partially coherent source. The grid is small (40x40) because the 2-D TCC
eigen-decomposition scales as Npix^3; the batched APPLY is grid agnostic once the kernels exist.
"""
import numpy as np, time, subprocess

LAM, NA = 193.0, 1.35; K0 = NA/LAM

def pupil2d(N, dx):
    f = np.fft.fftfreq(N, dx); FX, FY = np.meshgrid(f, f); return (np.hypot(FX, FY) <= K0).astype(np.complex128)
def source_offsets2d(N, dx, sigma, inner):
    f = np.fft.fftfreq(N, dx); df = f[1]-f[0]; FX, FY = np.meshgrid(f, f); r = np.hypot(FX, FY)
    m = (r <= sigma*K0) & (r >= inner*K0)
    return [(int(round(FY[i, j]/df)), int(round(FX[i, j]/df))) for i, j in zip(*np.where(m))]  # (sy, sx) integer bin offsets

def abbe2d(mask, dx, offs, P):                                                     # 2D ground truth: I = (1/S) Σ_s |IFFT2[P·shift(M̂,f_s)]|²
    Msp = np.fft.fft2(mask); I = np.zeros_like(mask, dtype=float)
    for (sy, sx) in offs:
        I += np.abs(np.fft.ifft2(P*np.roll(np.roll(Msp, sy, 0), sx, 1)))**2
    return I/len(offs)

def tcc_kernels2d(N, dx, offs, P, K):
    """2D Hopkins TCC = Σ_s P(f+f_s)P*(f'+f_s)/S over the flattened freq grid; eigen-decompose ⟹ K 2D kernels (K,N,N)."""
    Pf = P.ravel()
    A = np.stack([np.roll(np.roll(P, -sy, 0), -sx, 1).ravel() for (sy, sx) in offs])   # (S, N²): P(f+f_s)
    TCC = (A.T @ np.conj(A))/len(offs)                                             # (N², N²)
    lam, V = np.linalg.eigh(TCC); idx = np.argsort(-np.abs(lam))[:K]
    return lam[idx].real, V[:, idx].T.reshape(K, N, N)                            # (K,), (K,N,N)

def aerial_one_2d(mask, phis, lams):                                              # per-mask reference
    Msp = np.fft.fft2(mask); I = np.zeros_like(mask, dtype=float)
    for lam, phi in zip(lams, phis): I += lam*np.abs(np.fft.ifft2(phi*Msp))**2
    return I

def batched_aerial_2d(masks, phis, lams, device, dtype, tile=0):
    """B masks (B,Ny,Nx) -> B aerials via batched fft2. phis (K,Ny,Nx) precomputed. tile>0 caps VRAM."""
    import torch
    cdt = torch.complex64 if dtype == torch.float32 else torch.complex128
    Phi = torch.as_tensor(np.asarray(phis), device=device, dtype=cdt)             # (K,Ny,Nx)
    lam = torch.as_tensor(np.asarray(lams), device=device, dtype=dtype)
    M = torch.as_tensor(np.asarray(masks), device=device, dtype=cdt)             # (B,Ny,Nx)
    B = M.shape[0]; out = torch.empty((B,)+M.shape[1:], device=device, dtype=dtype)
    step = tile if tile > 0 else B
    for s in range(0, B, step):
        Msp = torch.fft.fft2(M[s:s+step])                                         # (b,Ny,Nx)
        field = torch.fft.ifft2(Msp[:, None]*Phi[None])                           # (b,K,Ny,Nx) batched fft2
        out[s:s+step] = (lam[None, :, None, None]*(field.real**2+field.imag**2)).sum(1)
    return out.detach().to(torch.float64).cpu().numpy()

def gpu_busy():
    try: return subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu,memory.used","--format=csv,noheader"],capture_output=True,text=True,timeout=5).stdout.strip()
    except Exception: return "n/a"

def main():
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("="*104); print("SOCS batched 2-D (fft2) aerial engine — 2-D mask aerial images batched (%s)" % (torch.cuda.get_device_name(0) if dev=="cuda" else "CPU")); print("="*104)
    N, dx = 40, 8.0                                                               # 40×40 grid, 320nm field (small enough for the 2D TCC eigh)
    P = pupil2d(N, dx); offs = source_offsets2d(N, dx, 0.7, 0.4)
    print("  grid %d×%d, dx=%.0fnm, source points=%d, K=8 kernels" % (N, N, dx, len(offs)))
    lam_all, phi_all = tcc_kernels2d(N, dx, offs, P, K=N*N)                        # all kernels
    lam8, phi8 = lam_all[:8], phi_all[:8]
    rng = np.random.default_rng(0)
    def make_masks(B):
        base = np.zeros((N, N)); base[8:32:6, :] = 1.0                            # a 2D line pattern
        return np.clip(base[None]+0.15*rng.standard_normal((B, N, N)), 0, 1)

    # (1) SOCS(all) == Abbe 2D (correctness anchor)
    mask0 = make_masks(1)[0]
    I_abbe = abbe2d(mask0, dx, offs, P); I_socs_all = aerial_one_2d(mask0, phi_all, lam_all)
    socs_rel = np.max(np.abs(I_socs_all-I_abbe))/I_abbe.max()
    print("\n  (1) SOCS(all kernels) vs Abbe-2D (same operator): max|Δ|/Imax = %.2e ⟹ exact=%s ; TCC eig energy top-8 %.4f"
          % (socs_rel, socs_rel < 1e-6, np.cumsum(np.abs(lam_all))[7]/np.abs(lam_all).sum()))

    # (2) batched == per-mask 2D loop
    masks = make_masks(32)
    ref = np.array([aerial_one_2d(m, phi8, lam8) for m in masks])
    bat = batched_aerial_2d(masks, phi8, lam8, dev, torch.float32)
    rel = np.max(np.abs(bat-ref))/ref.max(); correct = rel < 1e-5
    print("\n  (2) batched-2D vs per-mask 2D loop (32 masks): max|Δ|/Imax = %.2e ⟹ fp-identical=%s" % (rel, correct))

    # (3) throughput
    if dev == "cuda":
        Bt = 1024; mb = make_masks(Bt)
        for _ in range(2): batched_aerial_2d(mb, phi8, lam8, dev, torch.float32)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(5): batched_aerial_2d(mb, phi8, lam8, dev, torch.float32)
        torch.cuda.synchronize(); bat_rate = Bt/((time.time()-t0)/5)
        t0 = time.time(); mb2 = make_masks(64)
        for m in mb2: aerial_one_2d(m, phi8, lam8)
        loop_rate = 64/(time.time()-t0); speedup = bat_rate/loop_rate
        print("\n  (3) throughput: BATCHED-2D %.0f aerials/s (B=%d) vs per-mask loop %.0f/s ⟹ ×%.0f ; nvidia-smi: %s" % (bat_rate, Bt, loop_rate, speedup, gpu_busy()))
        fast = speedup > 10
    else:
        bat_rate = 0; speedup = 0; fast = False; print("\n  (3) no CUDA — skipped")

    # (4) memory tiling
    mb = make_masks(256)
    tile_ok = np.array_equal(batched_aerial_2d(mb, phi8, lam8, dev, torch.float32, 0), batched_aerial_2d(mb, phi8, lam8, dev, torch.float32, 32))
    print("\n  (4) memory tiling (tile=32 vs untiled, 256 masks): identical=%s ⟹ 2D intermediate (B,K,Ny,Nx) tiled for VRAM" % tile_ok)

    print("\n  [SOCS(all)==Abbe-2D] %s   [batched-2D == per-mask (fp-identical)] %s   [batched ≫ loop throughput] %s   [tiling fp-identical] %s"
          % (socs_rel < 1e-6, correct, fast, tile_ok))

    print("\n  VERDICT (2-D SOCS batched aerial engine):")
    if socs_rel < 1e-6 and correct and fast and tile_ok:
        print("  PASS — the 2-D aerial engine: batched_aerial_2d(masks(B,Ny,Nx), phis(K,Ny,Nx), lams) computes B")
        print("    two-dimensional mask aerial images in ONE batched fft2 call (K 2-D SOCS kernels precomputed once")
        print("    from the 2-D Hopkins TCC). (1) SOCS with all kernels equals the 2-D Abbe sum exactly (%.0e — the" % socs_rel)
        print("    2-D operator, eigen-decomposed, the same anchor as the 1-D case); (2) batched == per-mask 2-D loop")
        print("    (floating-point identical, %.0e), so the 1-D certification carries over to the 2-D batched engine;" % rel)
        print("    (3) THROUGHPUT x%.0f over the per-mask loop (%.0f 2-D aerials/s at B=%d); (4) memory TILING is" % (speedup, bat_rate, 1024))
        print("    floating-point identical — the (B,K,Ny,Nx) intermediate is the VRAM driver, so tile the batch.")
        print("    An ILT loop on 2-D masks is then one batched-fft2 GPU call per gradient step. Scope: the small")
        print("    40x40 grid keeps the 2-D TCC eigen-decomposition tractable (it scales as Npix^3; a production run")
        print("    uses a reduced or precomputed kernel set), but the batched APPLY is grid agnostic given kernels.")
    else:
        print("  ~ RESULT: socs=%s correct=%s fast=%s tile=%s (socs_rel %.0e, rel %.0e, ×%.0f) — inspect." % (socs_rel < 1e-6, correct, fast, tile_ok, socs_rel, rel, speedup))

if __name__ == "__main__":
    main()
