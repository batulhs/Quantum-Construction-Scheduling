"""
run_demo.py -- the whole pipeline on REAL construction projects.

    py run_demo.py                 laptop (Windows / macOS / Linux)
    python run_demo.py             on Colab with a GPU: uses the 26-qubit slice automatically
    py run_demo.py --benchmark     also prints the classical benchmark (CP-SAT, heuristic)

PART 1  Real project input        read 3 DSLIB projects, report every conversion, validate
PART 2  Bottleneck detection      GAP 1: which crews/equipment delay each project, ranked
PART 3  Multi-resource conflicts  GAP 3: joint bottlenecks, conflict graph, conflict cores
PART 4  Hybrid scheduling         the proposed system, with ADAPTIVE decomposition (GAP 2)
PART 5  Quantum core              the top conflict core, solved whole by annealing and QAOA
PART 6  Summary                   every check printed as PASS / FAIL

Outputs go to results/.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings

warnings.filterwarnings("ignore")

import pandas as pd

from src import visualize as viz
from src.bottleneck import conflict_analysis, pair_synergy, rank_bottlenecks
from src.classical import cpsat, serial_sgs
from src.dslib import load_selected, validate_project
from src.evaluate import SolverResult, build_report, is_feasible, makespan, utilisation, validate_schedule
from src.qubo import END, full_project_size
from src.quantum import aer_device, anneal, hybrid, qaoa
from src.slice_problem import EXHAUSTIVE_MAX_QUBITS, QUBIT_BUDGET, build_slice, exhaustive

HERE = os.path.dirname(os.path.abspath(__file__))
DATA, OUT = os.path.join(HERE, "data", "dslib"), os.path.join(HERE, "results")
CHECKS = []


def banner(t):
    print("\n" + "=" * 78 + "\n" + t + "\n" + "=" * 78)


def check(name, ok, detail=""):
    CHECKS.append(bool(ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))


def show(df):
    with pd.option_context("display.width", 220, "display.max_columns", 20):
        print(df.fillna("-").to_string(index=False))


def save_schedule(project, sched, path):
    rows = [{"id": a, "name": project.activities[a].name, "phase": project.activities[a].phase,
             "start_day": s, "finish_day": s + project.activities[a].duration}
            for a, s in sorted(sched.items(), key=lambda kv: (kv[1], kv[0]))]
    pd.DataFrame(rows).to_csv(path, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slice", choices=list(QUBIT_BUDGET), default=None,
                    help="quantum core size (default: gpu if a GPU is found, else laptop)")
    ap.add_argument("--benchmark", action="store_true",
                    help="also run the classical benchmark (CP-SAT, priority-rule heuristic)")
    args = ap.parse_args()
    device = aer_device()
    size = args.slice or ("gpu" if device == "GPU" else "laptop")
    os.makedirs(OUT, exist_ok=True)
    t_all = time.perf_counter()

    # ------------------------------------------------------------------ 1
    banner("PART 1  Real construction projects (DSLIB v3.4, Ghent University)")
    projects = load_selected(DATA)
    for spec, p, notes in projects:
        print(f"\n  {spec['code']}  {p.info['label']}: {p.n} activities, {len(p.edges())} links, "
              f"{len(p.capacity)} crews/equipment, calendar {p.info['calendar']}")
        print(f"    availability: {p.info['availability']}")
        print(f"    conversions: {dict(notes) or 'none'}")
        errs = validate_project(p)
        check(f"{p.info['label']}: data valid (no cycles, no impossible demand)", not errs, "; ".join(errs))

    # ------------------------------------------------------------------ 2 + 3
    analyses = {}
    banner("PART 2  GAP 1 - Resource bottleneck detection and ranking")
    for spec, p, _ in projects:
        tag = spec["code"]
        rk = rank_bottlenecks(p)
        print(f"\n--- {tag} {p.info['label']}: critical path {rk.attrs['critical_path']} days; "
              f"with crew/equipment limits {rk.attrs['makespan']} days ---")
        show(rk.head(6).assign(Resource=[viz.en(r) for r in rk.head(6)["Resource"]]))
        rk.to_csv(os.path.join(OUT, f"{tag}_bottlenecks.csv"), index=False)
        viz.bottleneck_chart(rk, f"{p.info['label']} ({tag}): resource bottlenecks",
                             os.path.join(OUT, f"{tag}_bottlenecks.png"))
        top = rk.iloc[0]
        check(f"{p.info['label']}: top bottleneck identified", top["Impact (days)"] > 0,
              f"{viz.en(top['Resource'])}, {top['Impact (days)']} days of delay")
        analyses[tag] = {"ranking": rk}

    banner("PART 3  GAP 3 - Multi-resource conflict detection")
    for spec, p, _ in projects:
        tag = spec["code"]
        rk = analyses[tag]["ranking"]
        an = conflict_analysis(p, rk)
        syn = pair_synergy(p, rk)
        analyses[tag].update(analysis=an)
        print(f"\n--- {tag} {p.info['label']} ---")
        print(f"  bottleneck resources: {[viz.en(r) for r in an['bottlenecks']]}")
        print(f"  conflict graph: {len({a for e in an['edges'] for a in e})} activities, {len(an['edges'])} conflicts")
        if an["multi"]:
            print(f"  multi-resource activities ({len(an['multi'])}): " +
                  "; ".join(f"{a} {p.activities[a].name[:28]} -> {', '.join(viz.en(r) for r in sorted(rs))}"
                            for a, rs in sorted(an["multi"].items())))
        else:
            print("  multi-resource activities: none (one dominant bottleneck)")
        if len(syn):
            print("  resource pairs (joint effect of relieving both):")
            show(syn.head(3).assign(**{"Resource pair": [" + ".join(viz.en(x) for x in r.split(" + "))
                                                         for r in syn.head(3)["Resource pair"]]}))
        for i, c in enumerate(an["cores"][:3], 1):
            print(f"  conflict core {i}: {len(c['activities'])} activities over "
                  f"{', '.join(viz.en(r) for r in c['resources'])} (impact {c['impact']} days)")
        rows = [{"core": i, "activities": " ".join(c["activities"]),
                 "resources": "; ".join(c["resources"]), "impact_days": c["impact"],
                 "multi_resource_activities": " ".join(c["multi_resource_activities"])}
                for i, c in enumerate(an["cores"], 1)]
        pd.DataFrame(rows).to_csv(os.path.join(OUT, f"{tag}_conflict_cores.csv"), index=False)
        check(f"{p.info['label']}: conflict cores found", len(an["cores"]) > 0, f"{len(an['cores'])} cores")

    # ------------------------------------------------------------------ 4
    banner("PART 4  Hybrid quantum-classical scheduling with adaptive decomposition (GAP 2)")
    summary, decomp = [], []
    for spec, p, _ in projects:
        tag, label = spec["code"], p.info["label"]
        rk = analyses[tag]["ranking"]
        weight = dict(zip(rk["Resource"], rk["Impact (days)"]))
        print(f"\n--- {tag} {label} ---")
        hyb = hybrid(p, decomposition="adaptive", bottleneck_weight=weight)
        fix = hybrid(p, decomposition="fixed")
        for h, name in ((fix, "fixed window"), (hyb, "adaptive")):
            decomp.append({"Project": f"{tag} {label}", "Decomposition": name,
                           "Makespan (days)": makespan(p, h.schedule),
                           "Sub-QUBOs solved": int(h.notes["Sub-QUBOs solved"]),
                           "Mean qubits/sub-QUBO": int(h.notes["Mean qubits/sub-QUBO"]),
                           "Max qubits/sub-QUBO": int(h.notes["Max qubits/sub-QUBO"]),
                           "Runtime (s)": round(h.runtime_s, 1)})
        C, C0 = makespan(p, hyb.schedule), int(hyb.notes["Warm-start makespan"])
        u = utilisation(p, hyb.schedule)
        size1 = full_project_size(p, C)
        print(f"  initial plan {C0} days -> hybrid {C} days  (-{C0 - C} days, -{100 * (C0 - C) / C0:.1f}%)")
        print(f"  {hyb.notes['Sub-QUBOs solved']} quantum sub-problems, average {hyb.notes['Mean qubits/sub-QUBO']} "
              f"and at most {hyb.notes['Max qubits/sub-QUBO']} qubits each "
              f"(one-piece formulation would need {size1['total_qubits']:,})")
        v = validate_schedule(p, hyb.schedule)
        check(f"{label}: hybrid schedule valid", sum(v.values()) == 0, str(v))
        check(f"{label}: hybrid improves or keeps the initial plan", C <= C0, f"{C0} -> {C} days")

        viz.gantt(p, hyb.schedule, f"{label} ({tag}): hybrid schedule, {C} days",
                  os.path.join(OUT, f"{tag}_gantt_hybrid.png"))
        viz.resource_profile(p, hyb.schedule, f"{label}: daily crew/equipment use (hybrid)",
                             os.path.join(OUT, f"{tag}_resources_hybrid.png"))
        save_schedule(p, hyb.schedule, os.path.join(OUT, f"{tag}_schedule_hybrid.csv"))
        summary.append({"Project": f"{tag} {label}", "Activities": p.n,
                        "Availability": "recorded" if spec["rs"] is None else f"RS={spec['rs']}",
                        "Top bottleneck": viz.en(rk.iloc[0]["Resource"]),
                        "Initial plan (days)": C0, "Hybrid (days)": C,
                        "Reduction (%)": round(100 * (C0 - C) / C0, 1),
                        "Avg utilisation (%)": round(100 * sum(u.values()) / len(u), 1),
                        "Sub-QUBOs": int(hyb.notes["Sub-QUBOs solved"]),
                        "Max qubits": int(hyb.notes["Max qubits/sub-QUBO"]),
                        "Violations": sum(v.values()), "Runtime (s)": round(hyb.runtime_s, 1)})

        if args.benchmark:
            cp, lb = cpsat(p)
            rep = build_report(p, [cp, serial_sgs(p), hyb], lower_bound=lb)
            print("  benchmark:")
            show(rep)
            rep.to_csv(os.path.join(OUT, f"{tag}_benchmark.csv"), index=False)

    summary = pd.DataFrame(summary)
    decomp = pd.DataFrame(decomp)
    print("\n  Hybrid results:")
    show(summary)
    print("\n  Decomposition strategy (GAP 2):")
    show(decomp)
    summary.to_csv(os.path.join(OUT, "summary_all_projects.csv"), index=False)
    decomp.to_csv(os.path.join(OUT, "decomposition_fixed_vs_adaptive.csv"), index=False)
    viz.decomposition_chart(decomp, os.path.join(OUT, "decomposition_fixed_vs_adaptive.png"))
    for proj in decomp["Project"].unique():
        f = decomp[(decomp.Project == proj) & (decomp.Decomposition == "fixed window")].iloc[0]
        a = decomp[(decomp.Project == proj) & (decomp.Decomposition == "adaptive")].iloc[0]
        check(f"{proj}: adaptive decomposition at least as good as fixed",
              a["Makespan (days)"] <= f["Makespan (days)"],
              f"{f['Makespan (days)']} -> {a['Makespan (days)']} days")

    # ------------------------------------------------------------------ 5
    spec0, main_p, _ = projects[0]
    banner(f"PART 5  Quantum core of {spec0['code']} {main_p.info['label']} ({size}, {device} simulator)")
    sp, q, info = build_slice(main_p, size)
    print(f"  Extracted automatically from conflict core 1 ({', '.join(viz.en(r) for r in info['resources'])}):")
    for a in sp.activities.values():
        print(f"    {a.id:>3}  {a.name[:44]:44s} {a.duration} day(s), {a.demand}")
    print(f"  Residual capacity for this group: {sp.capacity}")
    print(f"  QUBO: {q.num_vars} qubits ({q.num_slack} slack), penalty P = {q.penalty:g}, "
          f"{q.bqm.num_interactions} quadratic terms")

    _, s_opt = cpsat(sp)                              # verified optimum of the core
    results, rand = [], None
    if q.num_vars <= EXHAUSTIVE_MAX_QUBITS:
        t = time.perf_counter()
        ground, ground_e, rand = exhaustive(sp, q, s_opt)
        results.append(SolverResult(f"Exhaustive check, all 2^{q.num_vars} bitstrings", ground, time.perf_counter() - t))
    t = time.perf_counter()
    samp, _, _ = anneal(q, num_reads=300)
    sa_sched, _ = q.decode(samp)
    sa_sched.pop(END, None)
    results.append(SolverResult("Quantum annealing (simulated)", sa_sched, time.perf_counter() - t))
    print(f"  Tuning QAOA angles on the {device} simulator ...")
    qa, qstats = qaoa(q, sp, optimum=s_opt)
    results.append(qa)
    rep = build_report(sp, results)
    rep = rep.drop(columns=[c for c in ("Gap vs bound (%)",) if c in rep])
    show(rep)
    rep.to_csv(os.path.join(OUT, f"core_{size}_results.csv"), index=False)

    rows = {}
    if rand:
        rows["Random guessing"] = {"feasible": 100 * rand["feasible_share"], "optimal": 100 * rand["optimal_share"]}
    rows[f"QAOA p=1 ({device} sim)"] = {"feasible": 100 * qstats["feasible_share"], "optimal": 100 * qstats["optimal_share"]}
    pd.DataFrame(rows).T.to_csv(os.path.join(OUT, f"core_{size}_qaoa_vs_random.csv"))
    viz.qaoa_bars(rows, os.path.join(OUT, f"core_{size}_qaoa_vs_random.png"), q.num_vars)
    for k, v in rows.items():
        print(f"  {k:24s} valid {v['feasible']:.3f}%   optimal {v['optimal']:.4f}%")

    print("\n  Validation (quantum core):")
    if rand:
        check("QUBO lowest-energy bitstring is the optimal schedule (formulation correct)",
              is_feasible(sp, ground) and makespan(sp, ground) == s_opt, f"{makespan(sp, ground)} days")
    check("Quantum annealing reaches the optimum",
          len(sa_sched) == sp.n and is_feasible(sp, sa_sched) and makespan(sp, sa_sched) == s_opt)
    check("QAOA returns a valid schedule", bool(qa.schedule) and is_feasible(sp, qa.schedule))
    if rand:
        check("QAOA samples valid schedules more often than random guessing",
              qstats["feasible_share"] > rand["feasible_share"],
              f"{qstats['feasible_share'] / max(rand['feasible_share'], 1e-12):.0f}x")
    if qa.schedule:
        viz.gantt(sp, qa.schedule, f"Quantum core ({q.num_vars} qubits): schedule found by QAOA",
                  os.path.join(OUT, f"core_{size}_gantt_qaoa.png"))

    # ------------------------------------------------------------------ 6
    banner("PART 6  Summary")
    print(f"  Validation checks passed: {sum(CHECKS)}/{len(CHECKS)}")
    print(f"  Total runtime: {time.perf_counter() - t_all:.1f} s")
    print(f"  Results written to: {OUT}")


if __name__ == "__main__":
    main()
