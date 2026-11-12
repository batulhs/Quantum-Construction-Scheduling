"""
ui_jobs.py -- heavy quantum jobs for the web UI, run in a separate process.

Qiskit's compiled (Rust) code can crash when called from Streamlit's script
thread, so app.py runs QAOA here, in a fresh worker process. Inputs and
outputs are plain Python values so they cross the process boundary on every OS.
"""

from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp


def _qaoa_job(data_dir: str, code: str, caps: tuple, size: str) -> dict:
    from .classical import cpsat
    from .dslib import load_selected
    from .project import Project
    from .quantum import qaoa
    from .slice_problem import EXHAUSTIVE_MAX_QUBITS, build_slice, exhaustive

    p = {s["code"]: pr for s, pr, _ in load_selected(data_dir)}[code]
    if caps:
        cap = dict(p.capacity)
        cap.update(dict(caps))
        p = Project(p.name, p.activities, cap, dict(p.info), list(p.order))
    sp, q, _ = build_slice(p, size)
    _, opt = cpsat(sp)
    res, stats = qaoa(q, sp, optimum=opt)
    rand = None
    if q.num_vars <= EXHAUSTIVE_MAX_QUBITS:
        _, _, rand = exhaustive(sp, q, opt)
    keep = ("feasible_share", "optimal_share")
    return {"schedule": dict(res.schedule or {}), "runtime_s": float(res.runtime_s),
            "stats": {k: float(stats[k]) for k in keep},
            "random": {k: float(rand[k]) for k in keep} if rand else None}


def run_qaoa(data_dir: str, code: str, caps: tuple, size: str, timeout_s: int = 900) -> dict:
    with cf.ProcessPoolExecutor(max_workers=1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(_qaoa_job, data_dir, code, caps, size).result(timeout=timeout_s)
