"""DIFFRACTION GRATING — the diffraction orders EMERGE from the Fraunhofer transform of a periodic aperture and must land
at the grating-equation angles sinθ_m = mλ/d (the Clairaut/emergence lesson + geometric thinking: the period d sets the
angles, nothing in the code prescribes them). Fraunhofer far-field = the Fourier transform of the aperture |Â(f_x)|², with
angle sinθ = λ·f_x.

FALSIFICATION: (1) ★the bright orders sit at sinθ_m = mλ/d (m=0,±1,±2,…) — emergent from the FFT, matched to the grating
equation; (2) the order angles are LINEAR in m (the grating equation, the dispersion); (3) ★N-slit SHARPENING — the
principal-maximum width ∝ 1/N (the array factor; a real spectrometer's resolving power); (4) ★MISSING ORDERS — when the
slit width a = d/k, every k-th order is extinguished by the single-slit sinc envelope (a geometric selection rule).
Output: a figure of the far-field orders in artifacts/diffraction_grating.png next to this module.
Reference: the grating equation sinθ_m = mλ/d (Born & Wolf, Principles of Optics; Fraunhofer diffraction of a periodic aperture).
"""
import os
import sys
import numpy as np

ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts")


def far_field(N, d, a, lam, L=4096.0, npts=2 ** 16):
    """Fraunhofer far-field of N slits (period d, width a). Returns sinθ axis and intensity |Â|²."""
    x = (np.arange(npts) - npts / 2) * (L / npts); dx = L / npts
    A = np.zeros(npts)
    x0 = -(N - 1) / 2 * d                                       # centre the grating
    for k in range(N):
        A[np.abs(x - (x0 + k * d)) <= a / 2] = 1.0
    Ahat = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(A)))
    fx = np.fft.fftshift(np.fft.fftfreq(npts, dx))             # spatial frequency
    sin_t = lam * fx                                           # Fraunhofer: sinθ = λ f_x
    return sin_t, np.abs(Ahat) ** 2


def peaks(sin_t, I, thresh):
    m = (I[1:-1] > I[2:]) & (I[1:-1] > I[:-2]) & (I[1:-1] > thresh * I.max())
    return sin_t[1:-1][m], I[1:-1][m]


def main():
    print("=" * 84)
    print("DIFFRACTION GRATING — orders EMERGE at sinθ=mλ/d from the Fraunhofer transform of a periodic aperture")
    print("=" * 84)
    lam, d, a, N = 1.0, 20.0, 5.0, 12
    print(f"\n  grating: N={N} slits, period d={d}, slit width a={a}, λ={lam}")

    sin_t, I = far_field(N, d, a, lam)
    pk, pkI = peaks(sin_t, I, 0.04)
    pk = pk[np.abs(pk) < 0.18]                                 # orders within ±3λ/d
    orders_an = np.array([m * lam / d for m in range(-3, 4) if abs(m * lam / d) < 0.18])
    matched = np.array([pk[np.argmin(np.abs(pk - o))] for o in orders_an])
    e1 = np.max(np.abs(matched - orders_an)) / (lam / d); ok1 = e1 < 0.02
    print(f"\n  (1) ★orders at sinθ_m=mλ/d: emergent peaks {np.round(matched,4)}")
    print(f"      vs grating eq {np.round(orders_an,4)} (max Δ{e1*100:.1f}% of one order)  {'✓' if ok1 else 'FAIL'}")

    # (2) linearity in m (grating equation / dispersion)
    ms = np.round(orders_an / (lam / d)).astype(int)
    slope, b = np.polyfit(ms, matched, 1)
    e2 = abs(slope - lam / d) / (lam / d); ok2 = e2 < 0.01 and abs(b) < 1e-3
    print(f"  (2) order angle LINEAR in m: dsinθ/dm={slope:.5f} vs λ/d={lam/d:.5f} (Δ{e2*100:.1f}%, intercept {b:.1e})  {'✓' if ok2 else 'FAIL'}")

    # (3) ★N-slit sharpening: principal-max FWHM ∝ 1/N
    def fwhm0(Nv):
        st, Iv = far_field(Nv, d, a, lam); Iv = Iv / Iv.max()
        c = np.argmin(np.abs(st))                              # the m=0 order at sinθ=0
        l = c; r = c
        while Iv[l] > 0.5: l -= 1
        while Iv[r] > 0.5: r += 1
        return (st[r] - st[l])
    ws = np.array([fwhm0(Nv) for Nv in (6, 12, 24, 48)])
    prod = ws * np.array([6, 12, 24, 48])                      # FWHM·N should be ≈ constant
    ok3 = np.max(np.abs(prod - prod.mean())) / prod.mean() < 0.1
    print(f"  (3) ★N-slit SHARPENING: FWHM₀·N = {np.round(prod,4)} for N=[6,12,24,48] (≈const ⇒ width∝1/N, the resolving power)  {'✓' if ok3 else 'FAIL'}")

    # (4) ★MISSING ORDERS: a = d/4 ⇒ the 4th, 8th… orders vanish (single-slit sinc zero)
    sin_t4, I4 = far_field(N, d, d / 4, lam)
    i4 = np.argmin(np.abs(sin_t4 - 4 * lam / d)); i3 = np.argmin(np.abs(sin_t4 - 3 * lam / d))
    missing = I4[i4] < 0.02 * I4.max() and I4[i3] > 0.05 * I4.max()
    ok4 = missing
    print(f"  (4) ★MISSING ORDERS (a=d/4): order m=4 intensity {I4[i4]/I4.max():.3f} (≈0, extinguished) vs m=3 {I4[i3]/I4.max():.3f} (present)  {'✓' if ok4 else 'FAIL'}")

    rend = False
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 3.4), dpi=120)
        m = np.abs(sin_t) < 0.18
        ax.plot(sin_t[m], I[m] / I.max(), "b-", lw=0.7)
        for o in orders_an: ax.axvline(o, color="r", ls=":", lw=0.6)
        ax.set_xlabel("sinθ"); ax.set_ylabel("intensity"); ax.set_title(f"grating orders at sinθ=mλ/d (N={N})", fontsize=9)
        os.makedirs(ART, exist_ok=True)
        fig.tight_layout(); fig.savefig(os.path.join(ART, "diffraction_grating.png")); plt.close(fig); rend = True
    except Exception as ex:
        print(f"  (render skipped: {ex})")

    ok = ok1 and ok2 and ok3 and ok4
    print("\n" + "=" * 84)
    if ok:
        print("DIFFRACTION GRATING validated — the orders EMERGE at the grating-equation angles (non-tautological):")
        print(f"  • bright orders fall at sinθ=mλ/d ({e1*100:.0f}% of an order), linear in m — the period d sets the angles, nothing else;")
        print(f"  • ★N-slit sharpening (width∝1/N, the resolving power) and ★missing orders (a=d/4 kills m=4) — geometric selection rules.")
        print(f"  ⇒ spectrometry / structural colour / X-ray-Bragg analogue; the periodic-aperture anchor for optics & wave physics.")
    else:
        print(f"  (1)orders {ok1} (2)linear {ok2} (3)sharpening {ok3} (4)missing {ok4}. Report honestly; fix at source.")
    print("=" * 84)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
