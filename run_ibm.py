"""
run_ibm.py -- run the quantum core's QAOA circuit on IBM quantum hardware.

    py run_ibm.py --fake      offline rehearsal on FakeKingston, a local noisy
                              replica of IBM's ibm_kingston chip (no account,
                              slow: ~1 min for 200 shots)
    py run_ibm.py             the REAL ibm_kingston via the free Open Plan
                              (needs the one-time account setup in README.md)

How it saves your free minutes: the QAOA angles are tuned on the laptop
simulator first (free, unlimited). Only ONE final job -- the tuned circuit,
sampled `--shots` times -- is sent to the quantum computer, which uses a few
seconds of the 10 free minutes.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src.classical import cpsat
from src.dslib import load_selected
from src.quantum import analyse_counts, optimise_qaoa, sample_qaoa
from src.slice_problem import build_slice, exhaustive
from src import visualize as viz

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data", "dslib"), os.path.join(HERE, "results")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true", help="use the offline FakeKingston replica")
    ap.add_argument("--shots", type=int, default=None, help="default: 4000 real, 200 fake")
    ap.add_argument("--backend", default="ibm_kingston", help="IBM machine name (real mode)")
    args = ap.parse_args()
    shots = args.shots or (200 if args.fake else 4000)
    os.makedirs(OUT, exist_ok=True)

    from qiskit_ibm_runtime import SamplerV2
    if args.fake:
        from qiskit_ibm_runtime.fake_provider import FakeKingston
        backend, label = FakeKingston(), "FakeKingston (noisy replica)"
    else:
        from qiskit_ibm_runtime import QiskitRuntimeService
        service = QiskitRuntimeService()          # uses the saved account
        try:
            backend = service.backend(args.backend)
        except Exception:
            backend = service.least_busy(operational=True, simulator=False)
        label = f"{backend.name} (real IBM hardware)"

    project = load_selected(DATA)[0][1]          # DSLIB C2025-05 Education Service
    sp, q, _ = build_slice(project, "laptop")    # automatic conflict core, <=16 qubits
    _, s_opt = cpsat(sp)
    _, _, rand = exhaustive(sp, q, s_opt)

    print(f"Quantum core {list(sp.activities)} (conflict over {list(sp.capacity)}), QUBO: {q.num_vars} qubits. Optimum makespan: {s_opt} days.")
    print("Step 1/3  tuning QAOA angles on the laptop simulator ...")
    ansatz, theta, _ = optimise_qaoa(q, reps=1)

    print("Step 2/3  laptop simulator reference sample ...")
    sim_counts, _ = sample_qaoa(q, ansatz, theta, shots=20000)
    _, sim = analyse_counts(q, sim_counts, sp, s_opt)

    print(f"Step 3/3  sending ONE job ({shots} shots) to {label} ...")
    t = time.perf_counter()
    counts, compiled = sample_qaoa(q, ansatz, theta, sampler=SamplerV2(mode=backend),
                                   backend=backend, shots=shots)
    ops = compiled.count_ops()
    hw_sched, hw = analyse_counts(q, counts, sp, s_opt)
    print(f"          done in {time.perf_counter() - t:.1f} s; compiled circuit depth "
          f"{compiled.depth()}, two-qubit gates {ops.get('cz', 0) + ops.get('ecr', 0)}")

    rows = {"Random guessing": rand, "QAOA laptop simulator": sim, f"QAOA {backend.name}": hw}
    table = pd.DataFrame({k: {"valid samples (%)": round(100 * v["feasible_share"], 3),
                              "optimal samples (%)": round(100 * v["optimal_share"], 4)}
                          for k, v in rows.items()}).T
    print("\n" + table.to_string())
    print(f"\nBest schedule found on {backend.name}: {hw_sched} "
          f"(makespan {max(hw_sched[a] + sp.activities[a].duration for a in hw_sched) if hw_sched else 'none'})")
    print("Note: real hardware is noisy; a deep circuit (hundreds of gates) loses most of its\n"
          "signal. This run demonstrates the pipeline end to end on a quantum computer.")

    tag = "fake" if args.fake else "real"
    table.to_csv(os.path.join(OUT, f"ibm_{tag}_run.csv"))
    viz.qaoa_bars({k: {"feasible": 100 * v["feasible_share"], "optimal": 100 * v["optimal_share"]}
                   for k, v in rows.items()}, os.path.join(OUT, f"ibm_{tag}_run.png"))
    print(f"Saved results/ibm_{tag}_run.csv and .png")


if __name__ == "__main__":
    main()
