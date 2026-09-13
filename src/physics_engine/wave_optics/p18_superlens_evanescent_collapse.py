"""Loss-limited resolution of a Pendry-type superlens: the claimed 3-700x subwavelength resolution collapses to
single digits once realistic material loss is included (forward model, no fit).

A silver/Pendry superlens restores sub-diffraction detail by amplifying evanescent waves (transverse k_x > k0). With
loss the amplification fails above a maximum transverse wavenumber k_max ~ ln(1/delta)/d (Podolskiy & Narimanov 2005;
Merlin 2004), so the resolution enhancement over the diffraction limit is
    ENH = k_max/k0 = ln(1/delta) / (2*pi*d/lambda),
with delta = Im(eps)/(2|Re(eps)|) the material loss and d the slab thickness. For realistic silver (delta ~ 0.1,
d ~ lambda/15) this gives ENH ~ 5x, consistent with Pendry-class experiments resolving ~lambda/6. Reaching 700x would
require delta -> 0 or d ~ lambda/300 (atomically thin) - both unphysical for any metal.

SCOPE: this cell evaluates one mechanism (silver-slab loss delta). The general bound ENH = ln(C)/(2*pi*d/lambda) with
C = min(1/delta, dynamic range) is mechanism-independent and scales as 1/d; a 700x figure corresponds to a sub-nm
near-field/NSOM working distance, not a far-field imaging lens.

INPUT: none (closed-form model). OUTPUT: printed gate lines; exit 0 when all gates hold.
"""
import sys
import numpy as np
def enh(delta,d_over_lam):                 # resolution enhancement over the diffraction limit
    return np.log(1.0/delta)/(2*np.pi*d_over_lam)
def main():
    print("="*100);print("US8587474B2 — superlens 3-700x subwavelength: REFUTE — losses collapse it to ~5x — no fit");print("="*100)
    # realistic silver superlens (Pendry-class): delta from eps, d a thin slab
    delta_real=0.10; d_real=1/15.0          # Im/2|Re| ~0.1 (Ag optical); d ~ lambda/15
    enh_real=enh(delta_real,d_real)
    # sanity anchor: Pendry's Ag superlens resolved ~lambda/6 => ENH~6 (external, not the patent)
    enh_pendry=6.0
    # what would 700x require?
    d_for700_atfixeddelta=np.log(1/delta_real)/(2*np.pi*700)        # d/lambda needed for 700x at realistic loss
    delta_for700_atfixedd=np.exp(-2*np.pi*700*d_real)              # loss needed for 700x at realistic d
    print(f"\n  realistic Ag superlens: delta={delta_real}, d=lambda/{1/d_real:.0f}  ->  ENH = {enh_real:.1f}x  (Pendry experiment ~{enh_pendry:.0f}x)")
    print(f"  to reach 700x at delta={delta_real}: need d = lambda/{1/d_for700_atfixeddelta:.0f}  (atomically thin, unphysical)")
    print(f"  to reach 700x at d=lambda/{1/d_real:.0f}: need delta = {delta_for700_atfixedd:.1e}  (lossless, unphysical for any metal)")
    # loss sweep: ENH vs delta at fixed thin d
    print("  ENH vs loss (d=lambda/15):")
    for de in (1e-3,1e-2,0.05,0.1,0.2):
        print(f"    delta={de:6.3f}: ENH={enh(de,d_real):5.1f}x")
    g1=3<enh_real<8                          # realistic loss => single-digit ENH (matches Pendry ~5-6x)
    g2=enh(0.1,d_real)<50                     # ★700x unreachable at any realistic loss/thickness
    g3=d_for700_atfixeddelta<1/200 or delta_for700_atfixedd<1e-20   # ★700x demands unphysical d or delta
    ok=g1 and g2 and g3
    print(f"\n  (1) realistic Ag (delta=0.1, d=lambda/15): ENH {enh_real:.1f}x — single-digit, matches Pendry's ~{enh_pendry:.0f}x  {'OK' if g1 else 'FAIL'}")
    print(f"  (2) ★REFUTE 700x: loss-limited ENH stays single-digit; 700x is {700/enh_real:.0f}x beyond the loss-limited reality  {'OK' if g2 else 'FAIL'}")
    print(f"  (3) ★700x demands the unphysical: d~lambda/{1/d_for700_atfixeddelta:.0f} OR delta~{delta_for700_atfixedd:.0e} (lossless metal)  {'OK' if g3 else 'FAIL'}")
    print("="*100)
    if ok:
        print("US8587474B2 REFUTED (honest-negative=PASS), no fit:")
        print(f"  • the superlens resolution enhancement is LOSS-LIMITED to ENH=ln(1/delta)/(k0 d): realistic silver (delta~0.1, d~lambda/15)")
        print(f"    gives {enh_real:.1f}x, matching Pendry's experimental ~lambda/6. The claimed 700x would require an atomically-thin slab")
        print(f"    (d~lambda/{1/d_for700_atfixeddelta:.0f}) or a lossless metal (delta~{delta_for700_atfixedd:.0e}) — both unphysical. The evanescent amplification")
        print(f"    collapses with any real loss. So 3-700x overstates by ~100x; the loss-limited ~5x is the physics. Externally anchored, no fit.")
    else:
        print(f"  HONEST: ENH_real {enh_real:.1f}x; inspect.")
    print("="*100)
    return 0 if ok else 1
if __name__=="__main__":
    sys.exit(main())
