"""Actual reaction extents preserve inventories; frozen v1 remains the comparator."""
import hashlib
import itertools
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from physics_engine.electrochem import battery_ocv_soc_equilibrium as ocv
from physics_engine.electrochem import battery_capacity_fade as fade
from physics_engine.electrochem import battery_coupled_electro_thermal_runaway as thermal

COEFF = sorted(b for b, _ in fade.SYNTH_CELLS.values())
MID = (COEFF[0] + COEFF[-1]) / 2
NOMINAL = (0.0, MID, 1.0)
BOUNDS = ((-ocv.OCV_NOISE_V, ocv.OCV_NOISE_V), (COEFF[0], COEFF[-1]), (0.9, 1.1))


def capacity(cycles, coefficient):
    return fade.Q0_SYNTH - fade.A_SQRT * np.sqrt(cycles) - coefficient * cycles**2


def primitive(soc, offset):
    return (2.9 + offset)*soc + 0.95*soc**2 - 0.2*soc**3


def simulate(parameters=NOMINAL, *, cycles=100, resistance=1.0, chemical=True,
             max_dt=5.0, horizon=8000.0):
    offset, coefficient, cooling = parameters
    q = float(capacity(cycles, coefficient))
    if q <= 0 or resistance <= 0:
        raise ValueError("Invalid capacity or resistance")
    ambient = 298.15
    temperature = ambient
    soc = 1.0
    state = np.array([thermal.RXN[0][4], thermal.RXN[1][4], thermal.RXN[2][4], thermal.RXN[3][4], 0.0])
    initial_energy = 3600*q*primitive(soc, offset)
    electrical = chemistry = lost = 0.0
    time = 0.0; peak = temperature; event = None; count = 0
    max_balance = 0.0; valid = True
    digest = hashlib.sha256()
    while time < horizon and count < 200000:
        voltage = float(ocv.ocv_equilibrium(soc)) + offset
        current = voltage / resistance if soc > 0 and resistance < 1e8 else 0.0
        power, rates = thermal.chem_power(state, temperature) if chemical else (0.0, [0.0]*4)
        chemical_power = power * thermal.V_VOL
        cooling_power = thermal.HA * cooling * (temperature - ambient)
        dtemp = (voltage*current + chemical_power - cooling_power) / thermal.MCP
        dt = min(max_dt, horizon-time, 0.25/max(abs(dtemp),1e-12), 0.002/max(max(rates),1e-12))
        inventory = (state[0], state[1], 1-state[2], state[3])
        for amount, rate in zip(inventory, rates):
            if rate > 0:
                dt = min(dt, amount/rate)
        if current > 0:
            dt = min(dt, soc*3600*q/current)
        if not np.isfinite(dt) or dt <= 0:
            raise RuntimeError("Nonpositive integration step")
        next_soc = max(0.0, soc-current*dt/(3600*q))
        de = 3600*q*(primitive(soc,offset)-primitive(next_soc,offset))
        extents = np.minimum(np.array(rates)*dt, np.array(inventory))
        dc = sum(thermal.RXN[i][2]*thermal.RXN[i][3]*extents[i] for i in range(4))*thermal.V_VOL
        dl = cooling_power*dt
        temperature += (de+dc-dl)/thermal.MCP
        electrical += de; chemistry += dc; lost += dl
        state[0] -= extents[0]; state[1] -= extents[1]
        state[2] += extents[2]; state[3] -= extents[3]; state[4] += extents[1]
        soc = next_soc; time += dt; count += 1
        peak = max(peak, temperature)
        if temperature-ambient > 200 and event is None:
            event = time
        balance = thermal.MCP*(temperature-ambient) - (electrical+chemistry-lost)
        max_balance = max(max_balance,abs(balance)/max(1.0, initial_energy+chemistry))
        valid = valid and 0 <= soc <= 1 and bool(np.all(state[:4] >= 0)) and bool(np.all(state[:4] <= 1))
        snapshot = np.array([time, temperature, soc, *state, electrical, chemistry, lost],dtype=np.float64)
        valid = valid and bool(np.isfinite(snapshot).all())
        digest.update(snapshot.tobytes())
    return {"parameters": list(parameters), "capacity_Ah": q, "initial_energy_J": initial_energy,
            "peak_C": peak-273.15, "event_time_s": event, "runaway": bool(peak-ambient > 200),
            "event_consistent": bool((event is not None) == (peak-ambient > 200)),
            "final_soc": soc, "electrical_heat_J": electrical, "chemical_heat_J": chemistry,
            "cooling_J": lost, "sensible_energy_J": thermal.MCP*(temperature-ambient),
            "max_relative_balance_error": max_balance,
            "electrical_depletion_error_J": abs(initial_energy-3600*q*primitive(soc,offset)-electrical),
            "valid_states": bool(valid), "completed": bool(time >= horizon), "steps": count,
            "trajectory_sha256": digest.hexdigest()}


def measure():
    nominal = simulate()
    corners = [simulate(p) for p in itertools.product(*BOUNDS)]
    isolated = []; leave_out = []
    for stage in range(3):
        alone = []
        for value in BOUNDS[stage]:
            p = list(NOMINAL); p[stage] = value
            alone.append(simulate(tuple(p)))
        isolated.append(alone)
        other = [j for j in range(3) if j != stage]
        group = []
        for pair in itertools.product(*(BOUNDS[j] for j in other)):
            p = list(NOMINAL)
            for j, value in zip(other, pair):
                p[j] = value
            group.append(simulate(tuple(p)))
        leave_out.append(group)
    null = simulate(resistance=1e9, chemical=False)
    no_chem = simulate(chemical=False)
    fresh = simulate(cycles=0)
    refined = simulate(max_dt=2.5)
    span = lambda rows: max(x["peak_C"] for x in rows)-min(x["peak_C"] for x in rows)
    joint = span(corners)
    single = [span(rows) for rows in isolated]
    removed = [span(rows) for rows in leave_out]
    return {"nominal": nominal, "corners": corners, "isolated": isolated, "leave_one_out": leave_out,
            "null": null, "no_chemistry": no_chem, "fresh": fresh, "refined": refined,
            "joint_peak_span_C": joint, "stage_peak_spans_C": single, "linear_sum_C": sum(single),
            "leave_one_out_spans_C": removed,
            "largest_leave_one_out_effect": ("ocv", "fade", "cooling")[int(np.argmax(np.array(joint)-removed))],
            "peak_refinement_delta_C": abs(nominal["peak_C"]-refined["peak_C"])}


def main():
    a,b = measure(),measure()
    encoded = json.dumps(a,sort_keys=True,allow_nan=False).encode()
    all_runs = [a["nominal"],*a["corners"],a["null"],a["no_chemistry"],a["fresh"],a["refined"]]
    all_runs += [r for group in a["isolated"]+a["leave_one_out"] for r in group]
    gates = {"two_full_results_identical": encoded == json.dumps(b,sort_keys=True,allow_nan=False).encode(),
             "finite_bounded_complete": all(r["valid_states"] and r["completed"] for r in all_runs),
             "energy_balance": all(r["max_relative_balance_error"] <= 1e-10
                 and r["electrical_depletion_error_J"] / r["initial_energy_J"] <= 1e-10 for r in all_runs),
             "open_no_chemistry_null": a["null"]["peak_C"] == 25.0 and not a["null"]["runaway"],
             "fresh_capacity_larger": a["fresh"]["capacity_Ah"] > a["nominal"]["capacity_Ah"],
             "event_consistency": all(r["event_consistent"] for r in all_runs),
             "peak_time_refinement": a["peak_refinement_delta_C"] <= 1.0,
             "uncertainty_composition": a["joint_peak_span_C"] <= a["linear_sum_C"]}
    gates = {name: bool(value) for name, value in gates.items()}
    report = {"measurement": a, "sha256": hashlib.sha256(encoded).hexdigest(), "gates": gates,
              "scope": "synthetic OCV/fade/reaction laws; no empirical transfer or safety certificate"}
    (ROOT/"reports/battery_chain_inventory_v2.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"gates":gates,"nominal":a["nominal"],"joint_peak_span_C":a["joint_peak_span_C"],
                      "linear_sum_C":a["linear_sum_C"],"stage_peak_spans_C":a["stage_peak_spans_C"],
                      "largest_leave_one_out_effect":a["largest_leave_one_out_effect"],
                      "peak_refinement_delta_C":a["peak_refinement_delta_C"]},indent=2))
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
