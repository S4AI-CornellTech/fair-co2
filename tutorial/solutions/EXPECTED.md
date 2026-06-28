# Expected results — segment-6 Fair-CO2 tutorial

Reference numbers. The three attribution methods reproduce Fair-CO2's own algorithms
(`baseline_attribution` / `ground_truth_shapley_attribution` / `temporal_shapley` in
`Fair-CO2/monte-carlo-simulations/dynamic-demand/dynamic_demand_sim.py`), building on the hierarchical
Shapley in `Fair-CO2/forecast/emb_shapley_lib.py`. Recompute with `./tutorial.sh …` from
`walkthrough/06_fairco2/`.

## Stage 1 — the verdict (`make demo-fairco2`)
RUP deviates from the fair Shapley share by **80.3% avg / 279% worst**; Fair-CO2 → **19.1% / 54.8%**
(≈4.2–5.1×), at **~600,000×** less compute. (Fair-CO2's committed 10k-sim Monte-Carlo result.)

## Stage 2 — neighbor swing (`--swing <workload>`)
`llama`: isolated **0.139** vs with-spark **0.069** gCO2e → **2.02×** swing. `spark`: **2.06×**.
(Absolutes use `act_core`'s aligned SSD coefficient; the swing **ratio** is coefficient-independent.)

## Stage 3 — attribute one schedule (`--workloads exercises/workloads.json`, budget = R740 1523.1 kg)
Schedule: llama (cpu 40, runtime 10, start 0), spark (cpu 60, runtime 10, start 0), faiss (cpu 100,
runtime 2, start 4). Concurrent demand peaks at **200 cores** (slots 4–5, the faiss burst).

| job | RUP | Shapley (fair) | Fair-CO2 | RUP err | F-CO2 err |
|---|---|---|---|---|---|
| llama | **507.7** | **304.6** | **380.8** | 67% | 25% |
| spark | **761.5** | **456.9** | **571.2** | 67% | 25% |
| faiss (burst) | **253.8** | **761.5** | **571.2** | 67% | 25% |

The 2-slot faiss burst drives the peak the cluster is built for, so its fair share is the largest
(761 kg) — but RUP, billing by CPU×runtime, charges it the least (254 kg). Fair-CO2's cheap
approximation corrects it the right way (571 kg, ~25% off exact vs RUP's ~67%). Editing the schedule
(faiss `runtime`/`start`, a spark burst) moves all three.

> The methods are Fair-CO2's; Stage 1 is its committed Monte-Carlo output. No Monte-Carlo or optimizer
> is run in the hands-on.
