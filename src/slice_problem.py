"""
slice_problem.py -- the piece of a real project that a quantum computer can
solve whole, chosen AUTOMATICALLY from the project's worst conflict.

Pipeline (Gaps 1 and 3):
  1. rank_bottlenecks       which resources delay the project, by how much
  2. conflict_analysis      which activities compete for them (on one or
                            several resources at once) -> conflict cores
  3. extract_core           grow a group of activities from the top core, as
                            large as the qubit budget allows
  4. build the slice        the group, with
       - only the resources it is in conflict over,
       - RESIDUAL capacity: what is left of each resource on those days once
         the activities outside the group have taken their share (at least
         the group's largest single demand, at most the recorded capacity),
       - horizon = heuristic schedule length + 1 day, so the QUBO also
         contains valid-but-slower schedules for the solver to reject.

Qubit budgets: laptop 16 (also checked exhaustively, all 2^n bitstrings),
GPU 26 (Colab), so the QAOA state vector fits in memory.
"""

from __future__ import annotations

from typing import List, Tuple

import dimod
import numpy as np

from .bottleneck import _es_usage, conflict_analysis, extract_core, rank_bottlenecks
from .classical import best_sgs
from .evaluate import is_feasible, makespan
from .project import Project
from .qubo import END, Qubo, build_qubo

QUBIT_BUDGET = {"laptop": 16, "gpu": 26}
EXHAUSTIVE_MAX_QUBITS = 20


def slice_instance(project: Project, ids: List[str], resources: List[str]) -> Project:
    es, use = _es_usage(project)
    start = min(es[a] for a in ids)
    end = max(es[a] + project.activities[a].duration for a in ids)
    cap = {}
    for k in resources:
        others = np.zeros_like(use[k])
        for a in project.activities.values():
            if a.id not in ids and a.demand.get(k, 0):
                others[es[a.id]:es[a.id] + a.duration] += a.demand[k]
        need = max(project.activities[a].demand.get(k, 0) for a in ids)
        cap[k] = int(min(project.capacity[k], max(need, project.capacity[k] - others[start:end].max())))
    s = project.subproject(ids, "slice", cap)
    for a in s.activities.values():
        a.demand = {r: u for r, u in a.demand.items() if r in cap}
    return s


def slice_qubo(s: Project) -> Tuple[Qubo, int]:
    H = makespan(s, best_sgs(s)) + 1
    es, ls = s.earliest_starts(), s.latest_starts(H)
    windows = {a: list(range(es[a], ls[a] + 1)) for a in s.activities}
    residual = {r: np.full(H + 5, c) for r, c in s.capacity.items()}
    q = build_qubo(s, windows, residual, mode="makespan",
                   end_window=list(range(s.critical_path_length(), H + 1)))
    return q, H


def build_slice(project: Project, size: str = "laptop"):
    """Returns (slice project, QUBO, info dict)."""
    ranking = rank_bottlenecks(project)
    analysis = conflict_analysis(project, ranking)
    edges = analysis["edges"]
    rs_of = {}
    for (i, j), r in edges.items():
        rs_of.setdefault(i, set()).update(r)
        rs_of.setdefault(j, set()).update(r)

    def contested(ids):
        """Resources at least two group members are in conflict over."""
        count = {}
        for a in ids:
            for r in rs_of.get(a, ()):
                count[r] = count.get(r, 0) + 1
        return sorted(r for r, c in count.items() if c >= 2)

    def build(ids):
        rs = contested(ids)
        if not rs:
            return None
        s = slice_instance(project, ids, rs)
        conflict = makespan(s, best_sgs(s)) > s.critical_path_length()   # crews really delay the group
        return slice_qubo(s)[0], conflict

    ids, q = extract_core(project, analysis, QUBIT_BUDGET[size], build)
    rs = contested(ids)
    s = slice_instance(project, ids, rs)
    q, H = slice_qubo(s)
    info = {"ranking": ranking, "analysis": analysis, "resources": rs, "horizon": H}
    return s, q, info


def exhaustive(slice_project: Project, q: Qubo, optimum: int):
    """Score ALL 2^n bitstrings: the exact ground state, and how often uniform
    random guessing would hit a valid / optimal schedule."""
    if q.num_vars > EXHAUSTIVE_MAX_QUBITS:
        raise ValueError(f"{q.num_vars} qubits is too many to enumerate")
    ss = dimod.ExactSolver().sample(q.bqm)
    ground, _ = q.decode(ss.first.sample)
    ground.pop(END, None)
    feas = opt = 0
    for smp in ss.samples():
        sch, bad = q.decode(dict(smp))
        sch.pop(END, None)
        if not bad and len(sch) == slice_project.n and is_feasible(slice_project, sch):
            feas += 1
            opt += makespan(slice_project, sch) == optimum
    n = len(ss)
    return ground, float(ss.first.energy), {"feasible_share": feas / n, "optimal_share": opt / n, "states": n}
