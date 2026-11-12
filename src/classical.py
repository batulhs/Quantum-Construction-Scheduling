"""
classical.py -- Layer 2: classical solvers.

cpsat()       Exact classical BASELINE (Google OR-Tools CP-SAT): interval
              variables + AddCumulative (daily resource use <= capacity).
              Returns the best schedule and a proven LOWER BOUND on makespan,
              against which every method's optimality gap is measured.

serial_sgs()  Serial Schedule Generation Scheme with the LFT priority rule, the
              textbook RCPSP heuristic. Fast, always valid. It is also the
              warm start of the hybrid method.
"""

from __future__ import annotations

import time
from typing import Dict, Tuple

import numpy as np

from .evaluate import SolverResult
from .project import Project


def horizon(project: Project) -> int:
    """A safe upper bound on any schedule: everything one after another,
    plus room for positive lags."""
    lag = sum(max(0, dl) for _, _, dl in project.edges())
    return sum(a.duration for a in project.activities.values()) + lag + 1


def cpsat(project: Project, time_limit_s: float = 30.0, workers: int = 8) -> Tuple[SolverResult, int]:
    from ortools.sat.python import cp_model

    t0 = time.perf_counter()
    acts, H = project.activities, horizon(project)
    m = cp_model.CpModel()
    start, iv = {}, {}
    for a in acts.values():
        start[a.id] = m.NewIntVar(0, H, f"s_{a.id}")
        iv[a.id] = m.NewFixedSizeIntervalVar(start[a.id], a.duration, f"i_{a.id}")
    for p, s, dl in project.edges():
        m.Add(start[s] >= start[p] + dl)
    for r, cap in project.capacity.items():
        users = [a for a in acts.values() if a.demand.get(r, 0) > 0 and a.duration > 0]
        if users:
            m.AddCumulative([iv[a.id] for a in users], [a.demand[r] for a in users], cap)
    cmax = m.NewIntVar(0, H, "makespan")
    m.AddMaxEquality(cmax, [start[a.id] + a.duration for a in acts.values()])
    m.Minimize(cmax)

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.random_seed = 0
    status = solver.Solve(m)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError("CP-SAT found no schedule")
    sched = {a: int(solver.Value(start[a])) for a in acts}
    return (SolverResult("CP-SAT (classical exact)", sched, time.perf_counter() - t0,
                         {"Status": solver.StatusName(status)}),
            int(solver.BestObjectiveBound()))


def new_usage(project: Project) -> Dict[str, np.ndarray]:
    H = horizon(project) + max(a.duration for a in project.activities.values()) + 1
    return {r: np.zeros(H, dtype=int) for r in project.resources}


def earliest_feasible_start(project: Project, aid: str, schedule: Dict[str, int],
                            usage: Dict[str, np.ndarray]) -> int:
    """First day that satisfies every link from scheduled predecessors and on
    which the activity fits within remaining capacity for its whole duration."""
    a = project.activities[aid]
    t = max([0] + [schedule[p] + dl for p, dl in a.preds])
    while not all(np.all(usage[r][t:t + a.duration] + u <= project.capacity[r]) for r, u in a.demand.items()):
        t += 1
    return t


def place(project: Project, aid: str, s: int, schedule, usage) -> None:
    a = project.activities[aid]
    schedule[aid] = s
    for r, u in a.demand.items():
        usage[r][s:s + a.duration] += u


PRIORITY_RULES = ("LFT", "LST", "MTS", "GRPW")


def priority_order(project: Project, rule: str = "LFT"):
    """Precedence-respecting activity order. Among eligible activities, pick by:
    LFT  smallest latest finish time      LST  smallest latest start time
    MTS  most total successors            GRPW greatest rank positional weight
    (standard RCPSP priority rules, Kolisch 1996)."""
    acts = project.activities
    ls = project.latest_starts(project.critical_path_length())
    succ = project.successors()
    total_succ = {}
    for a in reversed(project.order):
        total_succ[a] = set(s for s, _ in succ[a]).union(*[total_succ[s] for s, _ in succ[a]])
    key = {
        "LFT": lambda a: ls[a] + acts[a].duration,
        "LST": lambda a: ls[a],
        "MTS": lambda a: -len(total_succ[a]),
        "GRPW": lambda a: -(acts[a].duration + sum(acts[s].duration for s, _ in succ[a])),
    }[rule]
    done, order = set(), []
    while len(order) < project.n:
        eligible = [a for a in acts if a not in done and all(p in done for p, _ in acts[a].preds)]
        nxt = min(eligible, key=lambda a: (key(a), a))
        order.append(nxt)
        done.add(nxt)
    return order


def serial_sgs(project: Project, rule: str = "LFT") -> SolverResult:
    t0 = time.perf_counter()
    schedule, usage = {}, new_usage(project)
    for aid in priority_order(project, rule):
        place(project, aid, earliest_feasible_start(project, aid, schedule, usage), schedule, usage)
    return SolverResult("Serial SGS (classical heuristic)", schedule, time.perf_counter() - t0)


def best_sgs(project: Project) -> Dict[str, int]:
    """Shortest SGS schedule over all priority rules (fast, always valid)."""
    best = None
    for rule in PRIORITY_RULES:
        s = serial_sgs(project, rule).schedule
        ms = max(s[a] + project.activities[a].duration for a in s)
        if best is None or ms < best[0]:
            best = (ms, s)
    return best[1]
