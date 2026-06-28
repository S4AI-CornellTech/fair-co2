# Expected results — Fair-CO2 tutorial

Shared budget: a 1,523.1 kg server. Three jobs over a 10-slot window — llama (40 cores, 10 slots),
spark (60, 10), faiss (100, 2, starting slot 4). Concurrent demand peaks at 200 cores in slots 4-5.

| job | RUP | Shapley | Fair-CO2 | RUP error |
|---|---|---|---|---|
| llama (LLM serving) | 507.7 | 304.6 | 380.8 | 67% |
| spark (batch ETL) | 761.5 | 456.9 | 571.2 | 67% |
| faiss (vector search) | 253.8 | 761.5 | 571.2 | 67% |

RUP charges faiss the least (it runs only briefly), but its fair Shapley share is the most — it is
what doubles the peak the server was sized for.

Variation — set faiss's `runtime` to 10 (no burst): RUP's error drops to 0%, i.e. RUP equals the fair
Shapley share (304.6 / 456.9 / 761.5 for llama / spark / faiss). With steady demand, proportional
billing is fair; the 67% error above comes entirely from faiss's burst. (Fair-CO2 stays a close
approximation: 338.5 / 507.7 / 676.9.)
