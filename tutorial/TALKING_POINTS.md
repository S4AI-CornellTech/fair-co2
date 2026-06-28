# Segment 6 — Fair-CO2: who *fairly* pays for the shared carbon? (THE CLIMAX)

**Story beat.** The payoff. The data center is built (segment 1, ACT) and running
(segment 5, EServe); its embodied + operational carbon now has to be **charged
back** to the software that shares the machine. Everyone in industry does this the
same way — split the carbon in proportion to resource use
(**Resource-Utilization-Proportional, RUP** = Google's operational carbon
accounting + the Green Software Foundation's SCI). Fair-CO2's punchline: **that
default is unfair**, and you can measure exactly how unfair against a
game-theoretic ground truth.

**Run it:** the 5-min demo runs from the **full-stack-carbon** suite (`make demo-fairco2`) — a direct
pandas reduction over Fair-CO2's committed Monte-Carlo results (`ref-sim-results/`, read-only),
reproducing the
paper's Figures 7-9. No cluster, no live workloads; the Fair-CO2 submodule is only
ever read.

---

### SETUP (the hook): your carbon bill depends on your neighbor

From Fair-CO2's embodied-carbon colocation matrix, the **same job's** attributed
embodied carbon swings ~**2x** purely depending on what it shares the node with:

| Job | sharing node w/ Spark | running alone | swing |
|---|---|---|---|
| Llama-3-8B | 0.069 gCO₂e | 0.139 gCO₂e | **2.02x** |
| Spark | 0.122 gCO₂e | 0.252 gCO₂e | **2.06x** |

If a job's bill can move 2x just from its neighbor, "how do we split it?" is not a
rounding question — it decides who pays. Figure: `fairco2_neighbor_swing.png`.
*(Absolute gCO₂e use `act_core`'s aligned SSD coefficient — the suite's storage
convergence — but the **swing is a ratio**, independent of the SSD/DRAM coefficients,
so the unfairness argument is unchanged.)*

### REVEAL (the climax): proportional splitting is ~80% wrong

Deviation from the **fair Shapley share** (the provably fair way to divide a shared
cost), averaged over 10,000 Monte-Carlo simulations:

| Method | Dynamic demand: avg / worst | Interference: avg / worst |
|---|---|---|
| **RUP** (industry default) | **80.3% / 279.0%** | **9.7% / 31.7%** |
| Demand-Proportional | 31.3% / 89.9% | — |
| **Fair-CO₂** | **19.1% / 54.8%** | **1.7% / 5.0%** |

RUP misattributes each job by **~80% on average — up to 279%** in the worst case.
Fair-CO2 cuts that error **~4-6x**. Figures: `fairco2_deviation.png` (the climax,
dynamic demand) and `fairco2_interference.png` (co-location interference).

---

**What to say (slide bullets):**

- **The baseline everyone ships is the one the paper discredits.** RUP / SCI looks
  reasonable (charge each job for the resources it used) but ignores that carbon is
  a *shared, non-separable* cost — exactly what Shapley values were invented to
  divide fairly. Measured that way, RUP is off by ~80%.
- **~600,000x less compute.** Exact Shapley is exponential (every coalition of
  co-located jobs), so nobody runs it live. Fair-CO2 *approximates* the fair share
  at ~600,000x less compute — cheap enough to attribute carbon **per job, in real
  time**, not in a post-hoc audit.
- **Fair + live attribution is what unlocks optimization (Section 8 denouement).**
  You cannot optimize what you misattribute by 80%: cheap, accurate, live shares
  turn carbon into a real signal a scheduler/placement engine can act on. The whole
  walkthrough — measure (ACT) → model (CarbonClarity/COFFEE/MicroGreen) → serve
  (EServe) → **attribute (Fair-CO2)** — exists to make this last step honest.
- **Reproduced from committed data, submodule untouched.** The numbers come from a
  pandas mean over `ref-sim-results/` (the same reduction Fair-CO2's own figure
  generators print); nothing is written into the Fair-CO2 repo.

**Then: hands-on (≈20 min, see `TUTORIAL.md`).** Participants attribute shared carbon themselves: read
the verdict (RUP ~80% vs Fair-CO2 ~19%) → see a job's bill swing ~2× by neighbor → **split the R740's
1,523 kg across one 3-job schedule three ways — RUP vs the fair Shapley share vs Fair-CO2's cheap
approximation** — and watch RUP *under-charge the brief burst that drives the peak* (which fairly owes
the most) while Fair-CO2 corrects it → then bill their own tenant schedule. The point to convey: this is
the carbon **ACT quantified and EServe provisioned, finally split fairly** — RUP gets it badly wrong, and
Fair-CO2 makes the fair split cheap enough to actually use.

**Closes the loop with:**

- This is the **accounting climax** for the carbon quantified upstream: the embodied
  budget being divided is **ACT's R740 server ≈ 1,523 kgCO₂e** (segment 1) and
  **EServe's H100 node ≈ 103 kgCO₂e** (segment 5). Fair-CO2 even hardcodes an
  **ACT-derived `ACT_cpu_chip_cf = 18,530 gCO₂e`** per CPU — the same kind of number
  segment 1 produces. `lib/verify_chain.py` checks those handoffs line up.
- **Prev — EServe (segment 5):** EServe allocates a slice of one accelerator's
  embodied carbon to a workload via `execution_time / lifetime`. Fair-CO2 answers
  the question EServe's single-tenant slice can't: when *many* jobs share that
  hardware at once, **what is each one's fair share?**
