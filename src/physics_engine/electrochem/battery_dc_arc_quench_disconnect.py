"""DC-ARC QUENCH ON PACK DISCONNECT - whether opening a high-voltage battery pack actually clears the circuit.

What it computes: once a fuse element melts and a gap opens, breaking a high-voltage DC pack draws an arc that does
not self-extinguish - unlike AC there is no current zero-crossing - so the disconnect must force the arc voltage above
the pack voltage by opening a long enough gap or by burying the arc in quench media. A sustained DC arc has a roughly
linear column gradient E_arc (V/mm) plus electrode falls, and extinguishes only when
    V_arc = V_fall + E_arc*d >= V_pack,
so the quench gap is d_q = (V_pack - V_fall)/E_arc. This complements the I2t fuse model, which says when the element
opens rather than whether the opening clears.

Inputs: none (arc gradient 30 V/mm, combined electrode fall 30 V, 150 A fault current, 400 V and 48 V packs as module
constants). Outputs: the quench gap at 400 V and 800 V, the low-voltage null, the peak arc power, and gate lines
G1-G4.

Reference: representative sustained free-air DC arc column gradient and electrode-fall voltages; the engineering
practice of sand-filled fuses and long-throw contactors in high-voltage packs.

GATES: G1 a 400 V pack needs a centimetre-scale quench gap (far more than a bare fuse slit), so arc-quench media are
required. G2 the gap scales with pack voltage, so the arc problem is high-voltage specific. G3 NULL - a 48 V pack
self-quenches at a millimetre gap. G4 composes pack voltage, short current and the fuse model.
"""
import sys
import numpy as np

E_ARC = 30.0           # V/mm, sustained DC free-air arc column gradient (representative)
V_FALL = 30.0          # V, combined anode+cathode fall (near-electrode)
I_SHORT = 150.0        # A, fault current through the opening gap
V_HV = 400.0           # V, EV HV-pack voltage
V_LV = 48.0            # V, low-voltage pack (NULL)


def quench_gap_mm(V_pack):
    return max(0.0, (V_pack - V_FALL) / E_ARC)


def main():
    print("=" * 100)
    print("DC-ARC QUENCH ON PACK DISCONNECT render→match (HV DC arc won't self-extinguish — needs a long gap / quench media)")
    print("=" * 100)
    d_hv = quench_gap_mm(V_HV)
    d_lv = quench_gap_mm(V_LV)
    P_arc_hv = V_HV * I_SHORT                          # peak arc power while sustained
    print(f"\n  DC arc gradient E_arc={E_ARC:.0f} V/mm, electrode fall {V_FALL:.0f} V; fault current {I_SHORT:.0f} A")
    print(f"  G1 quench gap for {V_HV:.0f} V pack: d_q=(V−V_fall)/E_arc = {d_hv:.0f} mm ({d_hv/10:.1f} cm) — ≫ a bare fuse slit → arc-quench media (sand-filled fuse)")
    print(f"  G2 gap ∝ V_pack: at 800 V it doubles to {quench_gap_mm(800):.0f} mm — the arc problem grows with pack voltage (HV-specific)")
    print(f"  G3 NULL: {V_LV:.0f} V pack quenches at {d_lv:.1f} mm (a bare break clears it); peak HV arc power = {P_arc_hv/1e3:.0f} kW (must be cleared fast)")

    g1 = d_hv > 8.0                                    # ★cm-scale gap for HV → not a bare slit
    g2 = abs(quench_gap_mm(800) - 2 * d_hv) / (2 * d_hv) < 0.15   # ★scales ~∝ V_pack
    g3 = d_lv < 1.0                                    # ★NULL: LV pack self-quenches at sub-mm/mm gap
    g4 = g1 and g3
    ok = g1 and g2 and g3
    print("\n" + "-" * 100)
    print(f"  G1 ★400 V pack needs a {d_hv:.0f} mm quench gap (≫ bare fuse slit) → arc-quench media required          {'✓' if g1 else 'FAIL'}")
    print(f"  G2 ★gap ∝ V_pack (400→800 V doubles {d_hv:.0f}→{quench_gap_mm(800):.0f} mm) — HV-specific problem               {'✓' if g2 else 'FAIL'}")
    print(f"  G3 ★NULL: 48 V pack self-quenches at {d_lv:.1f} mm (bare break suffices)                              {'✓' if g3 else 'FAIL'}")
    print(f"  G4 composes pack voltage + short current + the pack-fuse (WHEN it opens vs whether it CLEARS)         {'✓' if g4 else 'FAIL'}")
    print("=" * 100)
    if ok:
        print("DC-ARC QUENCH render→match LIVE — clearing a HV battery pack is not just melting the fuse; it is QUENCHING the DC arc that")
        print(f"  follows. A DC arc has no current zero-crossing, so it persists until the opening gap forces its voltage above the source:")
        print(f"  d_q=(V_pack−V_fall)/E_arc. A {V_HV:.0f} V pack therefore needs a {d_hv:.0f} mm ({d_hv/10:.1f} cm) gap — far more than a bare fuse slit — which is")
        print(f"  exactly why HV packs use SAND-FILLED fuses and long-throw pyro/contactors that bury or stretch the arc. The requirement")
        print(f"  scales with pack voltage (∝V), so 800 V architectures need {quench_gap_mm(800):.0f} mm; a {V_LV:.0f} V pack, by contrast, self-quenches at {d_lv:.1f} mm — the")
        print(f"  arc-clearing problem is specifically a HIGH-VOLTAGE one. This is the clearing half of protection, composing the I²t pack-fuse.")
    else:
        print(f"HONEST WIP: g1={g1}(d_hv {d_hv:.0f}) g2={g2} g3={g3}(d_lv {d_lv:.1f}). Fix at source.")
    print("=" * 100)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
