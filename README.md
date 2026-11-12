# Quantum Optimization of Construction Project Scheduling and Resource Allocation

A hybrid quantum-classical scheduler for **real construction projects** that

1. **finds the bottlenecks**: ranks every crew and piece of equipment by how many days it delays the project (Gap 1);
2. **finds multi-resource conflicts**: activities blocked by several resources at once, grouped into conflict cores (Gap 3);
3. **decomposes adaptively**: splits the project into quantum sub-problems sized to each conflict and to a qubit budget (Gap 2);
4. **solves the worst conflict whole** on a quantum algorithm (QAOA, simulated annealing), extracted automatically.

**Data:** DSLIB v3.4, the real-project database of Ghent University's OR&S group
(Batselier & Vanhoucke, 2015, *Int. J. Project Management* 33(3):697–710). Originals in `data/dslib/`.

| Project (DSLIB code) | Type | Activities | Crew availability |
|---|---|---|---|
| C2025-05 *Education Service* | school construction, Peru | 58 | **as recorded** |
| C2025-06 *Primary School* | school construction, Peru | 52 | reduced, resource strength 0.5 |
| C2012-02 *Nut Mixing Station* | industrial plant, Belgium (crane, excavator, aerial platform…) | 58 | reduced, resource strength 0.2 |

## Results at a glance (laptop run)

| Project | Top bottleneck | Initial plan | **Hybrid** | Reduction | Largest sub-problem |
|---|---|---|---|---|---|
| *Education Service* | painters (29 days of delay) | 76 days | **54 days** | **−29%** | 98 qubits |
| *Primary School* | plumbers (20 days) | 80 | 80 | 0% | 97 qubits |
| *Nut Mixing Station* | aerial platform + labourers (multi-resource) | 27 | 27 | 0% | 100 qubits |

- **Quantum core** (*Education Service*, 3 painter jobs, 16 qubits, chosen automatically): annealing and QAOA both return the optimal 4-day schedule.
- **QAOA** samples a valid schedule **35× more often** than random guessing.
- 22/22 validation checks pass, and all schedules have zero rule violations.

---

## 0. Interactive web app (new)

First do the one-time setup in section 1 (create `.venv`, install `requirements.txt`). Then:

```powershell
py -m streamlit run app.py
```

Your browser opens at <http://localhost:8501>. Everything on screen is computed live by the same code as
`run_demo.py`, on the real DSLIB projects.

| Tab | What you can do |
|---|---|
| **Overview** | key numbers for the chosen project, the 6-step pipeline, the latest full-run results |
| **① Bottlenecks** (Gap 1) | which crews delay the project and by how many days; interactive bars and table |
| **② Conflicts** (Gap 3) | interactive conflict graph (hover any activity), multi-resource jobs, conflict cores, resource pairs |
| **③ Hybrid schedule** (Gap 2) | press **Run**: hybrid quantum-classical schedule, zoomable Gantt chart, daily crew use vs capacity, CSV download |
| **④ Quantum core** | the worst conflict as a QUBO (heatmap); solve it with quantum annealing or QAOA; QAOA vs random guessing |
| **Method** | the QUBO formula, the three research gaps, data source |

**Sidebar controls**
- *Construction project*: switch between the three real projects.
- *What-if: crew sizes*: change the size of the top-4 bottleneck crews; every tab recomputes. For example, give
  *Education Service* one more painter and watch the bottleneck move.
- *Qubit budget*: the largest sub-problem the hybrid may send to the quantum solver (156 = IBM's free machine).

Stop the app with `Ctrl+C` in the terminal.

---

## 1. Run on your laptop (Windows, VS Code)

Needs **Python 3.10 or newer** (`py --version`).

1. Unzip, then in VS Code use **File → Open Folder** and pick the folder that directly contains `run_demo.py`.
   Windows often unzips into a folder inside a folder; open the inner one.
2. Open the terminal (`` Ctrl+` ``) and run one line at a time:

```powershell
py -m venv .venv
.\.venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r requirements.txt
py run_demo.py
py -m pytest -v
```

- `run_demo.py` takes about **1 minute** and should end with `Validation checks passed: 22/22`.
- `pytest` runs **39 tests** (including one that starts the web app) in about 1 minute.
- If activation says *"running scripts is disabled"*, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`
  once, then activate again.

**Options**

```powershell
py run_demo.py --benchmark       # also run the classical benchmark (CP-SAT, priority-rule heuristic)
py run_demo.py --slice gpu       # force the larger 26-qubit quantum core
```

## 2. Run on Google Colab with a GPU

The GPU speeds up the QAOA simulation, which lets the quantum core grow from **16 qubits** (laptop) to
**26 qubits**.

1. Go to <https://colab.research.google.com> → **File → Upload notebook** → choose `colab_gpu.ipynb`.
2. **Runtime → Change runtime type → T4 GPU → Save.**
3. Run the cells top to bottom. One cell asks you to upload this project zip, unmodified.
4. The last cell downloads all results as `results.zip`.

Expect several minutes; QAOA tuning on 26 qubits is the slowest step. The GPU path was prepared but could
not be tested on a GPU during development (see `TECHNICAL_DECISIONS.md` §8).

## 3. What you get in `results/`

| File | Shows |
|---|---|
| `<code>_bottlenecks.csv/.png` | **Gap 1**: every resource ranked by days of delay caused, utilisation, conflict days |
| `<code>_conflict_cores.csv` | **Gap 3**: groups of activities competing for the bottleneck resources |
| `decomposition_fixed_vs_adaptive.csv/.png` | **Gap 2**: schedule length and sub-problem sizes, fixed vs adaptive |
| `<code>_gantt_hybrid.png` | Gantt chart of the hybrid schedule |
| `<code>_resources_hybrid.png` | daily use of the busiest crews/equipment vs capacity |
| `<code>_schedule_hybrid.csv` | start/finish day of every activity |
| `summary_all_projects.csv` | one line per project |
| `core_<size>_results.csv`, `core_<size>_qaoa_vs_random.png`, `core_<size>_gantt_qaoa.png` | the quantum core |
| `<code>_benchmark.csv` | only with `--benchmark` |

(`<code>` = `C2025-05`, `C2025-06`, `C2012-02`.)

## 4. Optional: real IBM quantum hardware (free)

```powershell
py run_ibm.py --fake    # rehearsal on a noisy copy of IBM's ibm_kingston, no account (~1-2 min)
py run_ibm.py           # the real machine
```

One-time account setup:

1. Sign up at <https://quantum.cloud.ibm.com> and choose the free **Open Plan** (10 min of quantum time per 28 days).
2. Create an instance and copy your **API key** and the instance **CRN**.
3. Run once (this saves the key on your computer, not in the project):

```powershell
py -c "from qiskit_ibm_runtime import QiskitRuntimeService as S; S.save_account(token='YOUR_API_KEY', instance='YOUR_CRN', set_as_default=True)"
```

The script tunes the circuit on your computer and sends **one** job (a few seconds of quantum time). The job may
wait in IBM's shared queue first. Present the run as a demonstration that the pipeline works on real hardware.

## 5. Put the web app online (free)

See **`DEPLOY.md`** for step-by-step instructions (Streamlit Community Cloud, about 10 minutes, free, needs a GitHub account).

## 6. Project structure

```
data/dslib/*.xlsx      the three real DSLIB project files (unchanged originals)
src/project.py         data model; every link type -> one start-gap rule
src/dslib.py           reads DSLIB Excel files, resource-strength rule, project selection
src/bottleneck.py      GAP 1 bottleneck ranking, GAP 3 multi-resource conflicts and conflict cores
src/classical.py       priority-rule heuristic (initial plans); CP-SAT (verification, --benchmark)
src/qubo.py            QUBO formulation (variables, objective, penalties, slack)
src/quantum.py         annealing, QAOA (CPU or GPU), hybrid with GAP 2 adaptive decomposition
src/slice_problem.py   automatic quantum core from the top conflict core + exhaustive check
src/evaluate.py        one checker and one set of metrics for every schedule
src/visualize.py       charts
app.py                 interactive web app (Streamlit)
src/ui_jobs.py         runs QAOA for the web app in a separate process
run_demo.py            main entry point
run_ibm.py             optional IBM hardware run
colab_gpu.ipynb        Google Colab notebook
tests/test_all.py      39 tests
.streamlit/config.toml web-app theme
DEPLOY.md              how to put the web app online
TECHNICAL_DECISIONS.md methodology, decisions, results, limitations
```

## 7. Troubleshooting

- **`requirements.txt not found`**: the terminal is in the outer folder; `cd` into the inner one.
- **`py` not found**: use `python`, or reinstall Python with *Add to PATH* ticked.
- **`streamlit` not found**: run `py -m pip install -r requirements.txt` again inside the activated `.venv`, then use `py -m streamlit run app.py`.
- **Web app says the port is in use**: `py -m streamlit run app.py --server.port 8502`.
- **Colab shows `('CPU',)`**: the runtime has no GPU. Switch to T4 GPU and run again from the top.

## Citation for the data

Batselier, J. & Vanhoucke, M. (2015). Construction and evaluation framework for a real-life project database.
*International Journal of Project Management*, 33(3), 697–710. doi:10.1016/j.ijproman.2014.09.004
