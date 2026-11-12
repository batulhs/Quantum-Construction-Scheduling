"""
project.py -- Layer 1: the construction project data model.

A project is a set of ACTIVITIES. Each activity has:
    * a duration in working days,
    * precedence links to earlier activities,
    * how many units of each RESOURCE (crew or equipment) it uses per day.

Resources are "renewable": a crew is busy while an activity runs and free
again when it ends. Each resource has a daily capacity.

Precedence links -- one rule for every link type
------------------------------------------------
Real schedules (exported from ProTrack / MS Project) use four link types with
optional lags. Every one of them is a lower bound on the gap between two START
days, so we store each link as (predecessor, delta) meaning

        start[successor] >= start[predecessor] + delta

    finish-to-start  i FS+lag j :  delta = d_i + lag
    start-to-start   i SS+lag j :  delta = lag
    finish-to-finish i FF+lag j :  delta = d_i - d_j + lag
    start-to-finish  i SF+lag j :  delta = lag - d_j

delta may be negative (e.g. "FS-1d" lets j start a day before i ends).
This single form is what the classical solver, the checker and the QUBO use.

A schedule throughout this code is a dict {activity_id: start_day}.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class Activity:
    id: str
    name: str
    duration: int
    preds: List[Tuple[str, int]]          # (predecessor id, minimum start gap)
    demand: Dict[str, int]                # resource -> units per day
    phase: str = ""


def link_delta(kind: str, d_pred: int, d_succ: int, lag: int) -> int:
    """Convert a link of type FS/SS/FF/SF with a lag into a start-gap delta."""
    return {"FS": d_pred + lag, "SS": lag, "FF": d_pred - d_succ + lag, "SF": lag - d_succ}[kind]


@dataclass
class Project:
    name: str
    activities: Dict[str, Activity]
    capacity: Dict[str, int]
    info: Dict[str, str] = field(default_factory=dict)   # source, notes
    order: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.order:
            self.order = self.topological_order()

    @property
    def resources(self) -> List[str]:
        return list(self.capacity)

    @property
    def n(self) -> int:
        return len(self.activities)

    def edges(self) -> List[Tuple[str, str, int]]:
        """All links as (predecessor, successor, delta)."""
        return [(p, a.id, dl) for a in self.activities.values() for p, dl in a.preds]

    def successors(self) -> Dict[str, List[Tuple[str, int]]]:
        succ = {a: [] for a in self.activities}
        for p, s, dl in self.edges():
            succ[p].append((s, dl))
        return succ

    def topological_order(self) -> List[str]:
        indeg = {a: len(act.preds) for a, act in self.activities.items()}
        succ = self.successors()
        ready = sorted(a for a, d in indeg.items() if d == 0)
        out = []
        while ready:
            a = ready.pop(0)
            out.append(a)
            for s, _ in succ[a]:
                indeg[s] -= 1
                if indeg[s] == 0:
                    ready.append(s)
        if len(out) != self.n:
            raise ValueError("Precedence graph contains a cycle.")
        return out

    # ---- Critical Path Method (ignores resources) -----------------------
    def earliest_starts(self) -> Dict[str, int]:
        es = {}
        for a in self.order:
            es[a] = max([0] + [es[p] + dl for p, dl in self.activities[a].preds])
        return es

    def critical_path_length(self) -> int:
        es = self.earliest_starts()
        return max(es[a] + self.activities[a].duration for a in self.activities)

    def latest_starts(self, horizon: int) -> Dict[str, int]:
        """Latest start so the activity, and everything after it, still fits
        before `horizon`."""
        succ = self.successors()
        ls = {}
        for a in reversed(self.order):
            ls[a] = min([horizon - self.activities[a].duration] + [ls[s] - dl for s, dl in succ[a]])
        return ls

    def total_work(self, resource: str) -> int:
        """Resource-days of work required by the project (fixed by the data)."""
        return sum(a.duration * a.demand.get(resource, 0) for a in self.activities.values())

    def subproject(self, ids: List[str], name: str, capacity: Dict[str, int] = None) -> "Project":
        """A slice containing only `ids`; links to outside activities are dropped."""
        acts = {i: Activity(a.id, a.name, a.duration, [(p, dl) for p, dl in a.preds if p in ids],
                            dict(a.demand), a.phase)
                for i, a in self.activities.items() if i in ids}
        return Project(name, acts, dict(capacity or self.capacity), dict(self.info))
