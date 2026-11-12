"""
dslib.py -- reads real construction projects from DSLIB.

DSLIB (Dynamic Scheduling Library) is the empirical project database of the
Operations Research & Scheduling group, Ghent University:
    Batselier, J. & Vanhoucke, M. (2015). Construction and evaluation framework
    for a real-life project database. International Journal of Project
    Management, 33(3), 697-710. doi:10.1016/j.ijproman.2014.09.004
Release used: DSLIB v3.4 (231 projects). Each project is an Excel file exported
from the ProTrack scheduling software. We use three sheets:

    "Baseline Schedule"  one row per activity: ID, Name, WBS, Predecessors,
                         Duration, Resource Demand. Rows WITHOUT a duration are
                         WBS summary headings, not activities.
    "Resources"          Name, Type (Renewable/Consumable), Availability
    "Agenda"             working hours per day and working days per week

How each field is converted (all conversions are counted and reported)
-----------------------------------------------------------------------
Duration      "4d", "1d 3h", "6h"  -> working days, hours rounded UP to the
              next whole day (a part-day still occupies the crew that day).
Predecessors  "12FS", "6SS", "14FS-5d", "20FS+1w 2d" -> start-gap delta
              (see project.link_delta); weeks use the file's days-per-week.
Demand        "carpenter [2,00 #8]" -> 2 carpenters/day (comma is a decimal
              mark; the "#8" field is not a quantity and is ignored). A bare
              name means 1 unit. Fractional demands are rounded up.
Availability  "12 people", "1 mach" -> 12, 1. Consumable resources (materials
              priced per unit) are not crews/equipment and are skipped.
"""

from __future__ import annotations

import math
import os
import re
import warnings
from collections import Counter
from typing import Dict, List, Tuple

from .project import Activity, Project, link_delta

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

# ---------------------------------------------------------------------------
# The three real projects used in this study and why
# ---------------------------------------------------------------------------
# Selection (see TECHNICAL_DECISIONS.md section 2): construction sector,
# renewable resources recorded, 50-100 activities. CP-SAT showed that in 16 of
# 17 such projects the RECORDED crew limits never delay the project (optimum =
# critical path), so there is nothing to optimise. We therefore use:
#   * the one project whose recorded limits do bind -- used exactly as recorded
#   * two more whose availabilities are reduced with the standard
#     "resource strength" rule (Kolisch, Sprecher & Drexl 1995), clearly flagged
PROJECTS = [
    {"code": "C2025-05", "label": "Education Service", "rs": None},
    {"code": "C2025-06", "label": "Primary School", "rs": 0.5},
    {"code": "C2012-02", "label": "Nut Mixing Station", "rs": 0.2},
]


def _num(s) -> float:
    return float(str(s).replace(",", "."))


def find_file(folder: str, code: str) -> str:
    for f in sorted(os.listdir(folder)):
        if f.startswith(code) and f.endswith(".xlsx"):
            return os.path.join(folder, f)
    raise FileNotFoundError(f"No DSLIB Excel file for {code} in {folder}")


def read_dslib(path: str) -> Tuple[Project, Counter]:
    """Parse one DSLIB project workbook. Returns the project and a counter of
    every conversion that was not a plain 1:1 copy."""
    import openpyxl

    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    notes = Counter()

    agenda = list(wb["Agenda"].iter_rows(values_only=True))
    hpd = sum(1 for r in agenda[1:25] if r[0] and str(r[1]).strip().lower() == "yes") or 8
    dpw = sum(1 for r in agenda[1:8] if r[3] and str(r[4]).strip().lower() == "yes") or 5

    capacity: Dict[str, int] = {}
    for r in list(wb["Resources"].iter_rows(values_only=True))[2:]:
        if r[0] is None:
            continue
        if str(r[2]).strip() != "Renewable":
            notes["consumable resource skipped"] += 1
            continue
        m = re.match(r"\s*([\d.,]+)", str(r[3]))
        if m:
            capacity[str(r[1]).strip()] = int(_num(m.group(1)))

    rows = list(wb["Baseline Schedule"].iter_rows(values_only=True))[2:]
    headings = {str(r[2]): str(r[1]).strip() for r in rows if r[0] is not None and r[7] is None and r[2]}
    raw = {}
    for r in rows:
        if r[0] is None or r[7] is None:
            continue                                   # WBS heading, not an activity
        aid, name, wbs, preds_s, dur_s, dem_s = str(r[0]), str(r[1]).strip(), str(r[2] or ""), r[3], str(r[7]), r[8]

        m = re.fullmatch(r"\s*(?:([\d.,]+)d)?\s*(?:([\d.,]+)h)?\s*", dur_s)
        if m and (m.group(1) or m.group(2)):
            hours = _num(m.group(1) or 0) * hpd + _num(m.group(2) or 0)
        else:
            hours = _num(dur_s) * hpd
        dur = math.ceil(hours / hpd - 1e-9)
        if abs(hours / hpd - round(hours / hpd)) > 1e-9:
            notes["duration with hours, rounded up to whole days"] += 1

        links = []
        for p in str(preds_s or "").split(";"):
            p = p.strip()
            if not p:
                continue
            m = re.fullmatch(r"(\d+)(FS|SS|FF|SF)?([+-].*)?", p)
            if not m:
                notes["unreadable link skipped"] += 1
                continue
            kind, lag = m.group(2) or "FS", 0
            if m.group(3):
                days = sum(_num(q) * {"w": dpw, "d": 1, "h": 1 / hpd}[u]
                           for q, u in re.findall(r"([\d.,]+)\s*([wdh])", m.group(3)))
                lag = (-1 if m.group(3)[0] == "-" else 1) * math.ceil(days - 1e-9)
            if kind != "FS" or lag:
                notes[f"{kind} link{' with lag' if lag else ''}"] += 1
            links.append((m.group(1), kind, lag))

        demand = {}
        for p in str(dem_s or "").split(";"):
            p = p.strip()
            if not p:
                continue
            m = re.match(r"(.*?)\s*(?:\[\s*([\d.,]+)[^\]]*\])?\s*$", p)
            res, q = m.group(1).strip(), _num(m.group(2)) if m.group(2) else 1.0
            if res not in capacity:
                notes["demand on a non-renewable resource skipped"] += 1
                continue
            if q != int(q):
                notes["fractional demand rounded up"] += 1
            demand[res] = demand.get(res, 0) + math.ceil(q)

        phase = headings.get(wbs.rsplit(".", 1)[0], "") if "." in wbs else ""
        raw[aid] = (name, dur, links, demand, phase)

    acts = {}
    for aid, (name, dur, links, demand, phase) in raw.items():
        preds = []
        for p, kind, lag in links:
            if p not in raw:
                notes["link to a missing activity skipped"] += 1
                continue
            preds.append((p, link_delta(kind, raw[p][1], dur, lag)))
        acts[aid] = Activity(aid, name, dur, preds, demand, phase)

    used = {r: c for r, c in capacity.items() if any(r in a.demand for a in acts.values())}
    info = {"source": f"DSLIB v3.4, {os.path.basename(path)}",
            "calendar": f"{hpd} h/day, {dpw} days/week",
            "availability": "as recorded"}
    return Project(os.path.basename(path)[:-5], acts, used, info), notes


def resource_strength(project: Project, rs: float) -> Project:
    """Reduce availabilities with the resource-strength rule (Kolisch,
    Sprecher & Drexl 1995):

        capacity_k = r_min_k + round( RS * (r_max_k - r_min_k) )

    r_min_k = largest single-activity demand (below this, the project is
              impossible), r_max_k = peak daily use in the earliest-start
              schedule (at or above this, the resource never binds).
    RS = 1 -> never binds, RS = 0 -> scarcest feasible. Capacities are never
    raised above the recorded value. Activities, durations, links and demands
    are untouched."""
    es = project.earliest_starts()
    T = project.critical_path_length()
    cap = {}
    for r, rec in project.capacity.items():
        rmin = max(a.demand.get(r, 0) for a in project.activities.values())
        rmax = max(sum(a.demand.get(r, 0) for a in project.activities.values()
                       if es[a.id] <= t < es[a.id] + a.duration) for t in range(T))
        cap[r] = min(rec, rmin + int(round(rs * (rmax - rmin))))
    info = dict(project.info, availability=f"reduced by resource strength RS={rs} (recorded: {project.capacity})")
    return Project(project.name, project.activities, cap, info, list(project.order))


def load_selected(folder: str) -> List[Tuple[dict, Project, Counter]]:
    out = []
    for spec in PROJECTS:
        p, notes = read_dslib(find_file(folder, spec["code"]))
        if spec["rs"] is not None:
            p = resource_strength(p, spec["rs"])
        p.info["label"] = spec["label"]
        out.append((spec, p, notes))
    return out


def validate_project(project: Project) -> List[str]:
    """Problems that would make the instance meaningless or infeasible."""
    errors = []
    for a in project.activities.values():
        if a.duration < 0:
            errors.append(f"{a.id}: negative duration")
        for r, u in a.demand.items():
            if u > project.capacity.get(r, 0):
                errors.append(f"{a.id}: needs {u} {r}, capacity {project.capacity.get(r, 0)}")
    try:
        project.topological_order()
    except ValueError as e:
        errors.append(str(e))
    return errors
