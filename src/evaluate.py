"""
evaluate.py -- Layer 5: ONE validator and ONE set of metrics for every solver.

Every solver returns a schedule {activity_id: start_day}. This module checks it
and scores it the same way regardless of which solver produced it, so the
comparison is consistent (same problem, same constraints, same objective).

Metrics
-------
makespan        finish day of the last activity
optimality gap  (makespan - lower_bound) / lower_bound * 100, where the lower
                bound is CP-SAT's proven bound (equal to the optimum when
                CP-SAT proves optimality)
utilisation_k   (resource-days of work needing k) / (capacity_k * makespan).
                The numerator is fixed by the data, so a shorter makespan
                always means higher utilisation.
violations      missing activities + precedence violations +
                resource overload (excess units summed over all days)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .project import Project


def makespan(project: Project, schedule: Dict[str, int]) -> int:
    return max(schedule[a] + project.activities[a].duration for a in schedule) if schedule else 0


def usage_profile(project: Project, schedule: Dict[str, int]) -> Dict[str, np.ndarray]:
    """Units of each resource in use on each day."""
    T = makespan(project, schedule)
    prof = {r: np.zeros(T + 1, dtype=int) for r in project.resources}
    for a, s in schedule.items():
        act = project.activities[a]
        for r, u in act.demand.items():
            prof[r][s:s + act.duration] += u
    return prof


def validate_schedule(project: Project, schedule: Dict[str, int]) -> Dict[str, int]:
    """Count every kind of constraint violation (all zeros = valid schedule)."""
    missing = sum(1 for a in project.activities if a not in schedule)
    prec = 0
    for p, s, dl in project.edges():
        if p in schedule and s in schedule:
            if schedule[s] < schedule[p] + dl:
                prec += 1
    overload = 0
    negative = sum(1 for s in schedule.values() if s < 0)
    for r, prof in usage_profile(project, schedule).items():
        overload += int(np.clip(prof - project.capacity[r], 0, None).sum())
    return {"missing": missing, "precedence": prec,
            "resource_overload": overload, "negative_start": negative}


def is_feasible(project: Project, schedule: Dict[str, int]) -> bool:
    return sum(validate_schedule(project, schedule).values()) == 0


def utilisation(project: Project, schedule: Dict[str, int]) -> Dict[str, float]:
    C = makespan(project, schedule)
    return {r: project.total_work(r) / (project.capacity[r] * C) if C else 0.0
            for r in project.resources}


@dataclass
class SolverResult:
    method: str
    schedule: Dict[str, int]
    runtime_s: float
    notes: Dict[str, str] = field(default_factory=dict)


def build_report(project: Project, results: List[SolverResult],
                 lower_bound: Optional[int] = None) -> pd.DataFrame:
    rows = []
    for r in results:
        v = validate_schedule(project, r.schedule)
        C = makespan(project, r.schedule)
        u = utilisation(project, r.schedule)
        row = {
            "Method": r.method,
            "Makespan (days)": C,
            "Gap vs bound (%)": round(100.0 * (C - lower_bound) / lower_bound, 2) if lower_bound else None,
            "Runtime (s)": round(r.runtime_s, 3),
            "Avg utilisation (%)": round(100.0 * float(np.mean(list(u.values()))), 1),
            "Busiest resource util. (%)": round(100.0 * max(u.values()), 1),
            "Violations": sum(v.values()),
            "Feasible": "Yes" if sum(v.values()) == 0 else "No",
        }
        for k, val in r.notes.items():
            row[k] = val
        rows.append(row)
    return pd.DataFrame(rows)
