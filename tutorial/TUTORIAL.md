# Fair-CO2 — hands-on: split shared carbon fairly (≈20 min)

The data center is built (**ACT**) and provisioned (**EServe**) — the upstream steps of the
full-stack-carbon walkthrough. Now its embodied carbon has to be **charged back** to the software that
shares the machine. You'll see why the industry's proportional split is unfair, then attribute a shared
server's carbon yourself — the fair way — using Fair-CO2's own Shapley code. This follows the 5-minute
Fair-CO2 intro (`TALKING_POINTS.md`).

The helper:
```bash
cd Fair-CO2/tutorial
./tutorial.sh --workloads exercises/workloads.json   # proportional (RUP) vs fair (Shapley) split
./tutorial.sh --swing llama                          # "your bill depends on your neighbor"
```
Prereq: a Python env with pandas + numpy. Standalone, point the runner at it with
`PYTHON=/path/to/venv/bin/python ./tutorial.sh …` (or activate it so `python3` works). From the
**full-stack-carbon** suite, `make setup` builds it and `make tutorial-fairco2` runs these. Paths below
are relative to `Fair-CO2/tutorial/`.

**Three ways to bill shared carbon.** **RUP** (Resource-Utilization-Proportional — the industry default,
= Google's operational accounting + the Green Software Foundation SCI) charges each job in proportion to
its **resource-time** (CPU × runtime). **Shapley** — the provably fair division of a shared cost —
charges each job by its **marginal contribution to the peak** the hardware was built (and embodied) to
serve, but it's exponentially expensive. **Fair-CO2** *approximates* that fair Shapley share cheaply
enough (~600,000×) to bill per-job, live. We'll see all three on one schedule.

---

## Stage 1 — Read the verdict (≈3 min)

From the **full-stack-carbon** suite: `make demo-fairco2`. (Standalone, these headline numbers come
straight from Fair-CO2's committed 10,000-sim results under `../monte-carlo-simulations/*/ref-sim-results/`.)

Over those results: **RUP misattributes each job by ~80% on average
(up to 279%)** vs the fair Shapley ground truth; **Fair-CO2 cuts that to ~19% (≈4.2×)** — at **~600,000×
less compute** than computing Shapley exactly, which is what makes fair, per-job attribution tractable
at all. The industry-default split isn't slightly off; it's badly unfair.

---

## Stage 2 — The hook: your bill depends on your neighbor (≈4 min)

```bash
./tutorial.sh --swing llama
```
The same Llama-3-8B serving job's attributed embodied carbon **swings ~2× (2.02×)** just from *who it
shares the node with* — **0.139 gCO2e alone** vs **0.069 gCO2e next to Spark** (it splits the shared box
with a neighbor). Try `--swing spark` (2.06×). **Lesson:** a fixed "your share" number can't be right if
it changes depending on your neighbors — attribution has to be *relational*, which proportional billing
can't capture.

---

## Stage 3 — Attribute a shared node yourself (≈8 min)

Three jobs co-locate on one server for a 10-slot window and share its embodied carbon — ACT's **R740 =
1,523 kg** from segment 1. Open `exercises/workloads.json`: each job is a **schedule** entry — `cpu`
(cores), `runtime` (slots), `start` (slot). The cluster is built for the **peak of concurrent demand**.
Run it:

```bash
./tutorial.sh --workloads exercises/workloads.json
```
| job | RUP | Shapley (fair) | Fair-CO2 | RUP err | F-CO2 err |
|---|---|---|---|---|---|
| llama (serving, steady) | 507.7 | 304.6 | 380.8 | 67% | 25% |
| spark (batch, steady) | 761.5 | 456.9 | 571.2 | 67% | 25% |
| **faiss (index rebuild, 2-slot burst)** | **253.8** | **761.5** | **571.2** | **67%** | 25% |

Demand peaks at **200 cores** in slots 4–5: the `faiss` index rebuild **doubles** the cluster's peak for
two slots. That burst is what forces you to provision — and embody — the extra hardware, so **fairly it
owes the most (761 kg, 50%)**. But RUP bills by CPU×runtime, so it charges faiss the *least* (254 kg) and
over-charges the steady all-day batch (spark). **RUP is backwards** — off by 67%. **Fair-CO2** moves
faiss the right way (254 → 571 kg), landing ~25% from the exact fair share — far closer than RUP, and
cheap enough to compute live (Stage 1's 600,000×).

**Try this — the experiment that nails *why* RUP fails.** Change faiss's `runtime` from `2` to `10` so
it runs flat-out the whole window instead of as a burst, and re-run:

```bash
./tutorial.sh --workloads exercises/workloads.json
```
**All three methods collapse onto the same split** (RUP error → 0%):

| job | RUP | Shapley (fair) | Fair-CO2 |
|---|---|---|---|
| llama | 304.6 | 304.6 | 304.6 |
| spark | 456.9 | 456.9 | 456.9 |
| faiss (now always-on) | 761.5 | 761.5 | 761.5 |

When every job runs steadily, proportional billing **is** fair — RUP's ~80% error comes *entirely* from
**temporal, bursty demand**. Real workloads (LLM request spikes, nightly index rebuilds, batch bursts)
are exactly that, which is when Fair-CO2 earns its keep. (Other knobs to try: shift faiss's `start` off
the peak, or give spark its own burst.)

---

## Stage 4 — Capstone: fairly bill your tenants (≈4 min)

Copy `exercises/workloads.json`, put in **your own** tenant schedule (each tenant's `cpu`, `runtime`,
`start`) and the shared budget (`--budget <kg>`, or keep the upstream R740 + H100 = 1,626 kg), and run:

```bash
./tutorial.sh --workloads exercises/workloads.json --budget 1626
```
The helper reports each tenant's **RUP vs fair (Shapley) vs Fair-CO2** share and how far RUP misses.
That's the whole suite closing the loop: the carbon **ACT quantified** and **EServe provisioned**,
finally **split fairly** across the jobs that share the machine — and Fair-CO2 is what makes the fair
split cheap enough to actually bill on.

---

## Honesty (say it aloud)

The three methods reproduce Fair-CO2's own algorithms (`baseline_attribution` / exact
`ground_truth_shapley_attribution` / `temporal_shapley` from its `dynamic_demand_sim.py`), and the
hierarchical Shapley they build on is imported straight from `../forecast/emb_shapley_lib.py` — on
one editable schedule, no Monte-Carlo. The Stage-1 headline (RUP 80% vs Fair-CO2 19%) is Fair-CO2's
committed 10,000-sim result; that simulation and any downstream optimizer are not re-run here. The point:
**proportional billing of shared carbon is unfair; the fair Shapley share fixes it; and Fair-CO2 makes
that fair share cheap enough to compute live.**
