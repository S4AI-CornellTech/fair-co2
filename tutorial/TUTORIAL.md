# Fair-CO2 hands-on: attribute a shared server's carbon

A server's embodied carbon has to be divided among the jobs that share it. In this exercise you
attribute one server's carbon across a schedule of co-located jobs and compare three ways of doing
it: the common proportional split (RUP), the exact fair split (Shapley), and Fair-CO2's cheap
approximation of the fair split.

## Setup

Run from this folder, with a Python that has `pandas` and `numpy`. The full-stack-carbon suite
builds one at `.envs/fair-co2`:

```bash
cd Fair-CO2/tutorial
source ../../.envs/fair-co2/bin/activate     # or any env with pandas + numpy
python tutorial.py
```

`exercises/workloads.json` is the default schedule; pass `--workloads <file>` to use your own.

## The schedule

`exercises/workloads.json` has three jobs sharing a 1,523 kg server over a 10-slot window:

| job | cpu (cores) | runtime (slots) | start |
|---|---|---|---|
| llama (LLM serving) | 40 | 10 | 0 |
| spark (batch ETL) | 60 | 10 | 0 |
| faiss (vector search) | 100 | 2 | 4 |

llama and spark run the whole window. faiss runs for two slots in the middle. Concurrent demand sits
at 100 cores most of the time but jumps to 200 in slots 4-5 when faiss runs, so the server has to be
provisioned for a 200-core peak.

## What it prints

```
  job                             RUP    Shapley   Fair-CO2  RUP err Fair-CO2 err
  llama (LLM serving)         507.7kg    304.6kg    380.8kg      67%          25%
  spark (batch ETL)           761.5kg    456.9kg    571.2kg      67%          25%
  faiss (vector search)       253.8kg    761.5kg    571.2kg      67%          25%
```

- **RUP** splits the budget by CPU x runtime. faiss runs only two slots, so RUP charges it the least
  (254 kg).
- **Shapley** charges each job by how much it adds to the peak the server was built for. faiss is
  what pushes demand from 100 to 200 cores, so its fair share is the largest (762 kg) — about three
  times what RUP charges it. RUP gets the ranking backwards, and is off by 67%.
- **Fair-CO2** approximates the Shapley share (faiss 571 kg) — 25% off the fair share here, versus
  RUP's 67% — and is cheap enough to compute for every job.

## Change the schedule and re-run

Edit `exercises/workloads.json` and run `python tutorial.py` again — move faiss's `start` away from
slots 4-5, or give llama or spark its own short burst, and watch which job the fair share charges most.

## Notes

The three methods call Fair-CO2's own code — `baseline_attribution`, `ground_truth_shapley_attribution`,
and `temporal_shapley` in `monte-carlo-simulations/dynamic-demand/dynamic_demand_sim.py`, built on the
hierarchical Shapley in `forecast/emb_shapley_lib.py`. Use `--budget <kg>` to change the shared budget,
or `--fig` to also write a bar chart to `figures/`.
