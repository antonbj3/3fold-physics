"""Measure existing stage seams before designing a synthetic battery chain."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from physics_engine.electrochem import battery_ocv_soc_equilibrium as ocv
from physics_engine.electrochem import battery_capacity_fade as fade
from physics_engine.electrochem import battery_coupled_electro_thermal_runaway as thermal


def measure():
    soc = np.array([0.0, 0.5, 1.0])
    voltage = ocv.ocv_equilibrium(soc)
    thermal_voltage = thermal.V0 * (0.5 + 0.5 * soc)
    coefficients = sorted(b for b, _ in fade.SYNTH_CELLS.values())
    fade_rows = []
    for n in (0, 50, 100):
        q = np.array([fade.Q0_SYNTH - fade.A_SQRT * np.sqrt(n) - b * n*n for b in coefficients])
        fade_rows.append({"cycle": n, "capacity_min_Ah": float(q.min()),
                          "capacity_max_Ah": float(q.max()),
                          "retention_min": float(q.min()/fade.Q0_SYNTH),
                          "retention_max": float(q.max()/fade.Q0_SYNTH)})
    cases = []
    for resistance, chemical in ((1.0,True), (2.0,True), (3.0,True), (1.0,False), (1e9,True)):
        peak, flag, time = thermal.simulate(resistance, chem_on=chemical)
        cases.append({"resistance_ohm": resistance, "chemistry_on": chemical,
                      "peak_C": float(peak), "returned_runaway": bool(flag),
                      "rise_above_200": bool(peak-25.0 > 200),
                      "event_time": None if time is None else float(time)})
    return {"soc": soc.tolist(), "ocv_V": voltage.tolist(),
            "thermal_voltage_V": thermal_voltage.tolist(),
            "voltage_delta_V": (voltage-thermal_voltage).tolist(),
            "fade_noiseless_law": fade_rows,
            "thermal_cases": cases,
            "thermal_stored_energy_J": thermal.E_ELEC,
            "ocv_fresh_capacity_energy_J": float(3600*fade.Q0_SYNTH*(2.9+1.9/2-0.6/3)),
            "returned_flag_disagreements": sum(c["returned_runaway"] != c["rise_above_200"] for c in cases)}


def main():
    a,b = measure(),measure()
    encoded = json.dumps(a,sort_keys=True,allow_nan=False).encode()
    gates = {"two_complete_measurements_identical": encoded == json.dumps(b,sort_keys=True,allow_nan=False).encode(),
             "ocv_monotonic_and_in_original_window": all(ocv.LI_MIN <= v <= ocv.LI_MAX for v in a["ocv_V"])
                 and all(x < y for x,y in zip(a["ocv_V"],a["ocv_V"][1:])),
             "fresh_fade_retention_one": a["fade_noiseless_law"][0]["retention_min"] == 1.0,
             "thermal_flag_matches_its_documented_rise": a["returned_flag_disagreements"] == 0}
    report = {"measurement": a, "sha256": hashlib.sha256(encoded).hexdigest(), "gates": gates,
              "scope": "synthetic stage laws; no transfer certificate between measured populations"}
    output = ROOT / "reports/battery_chain_mechanism_probe.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
