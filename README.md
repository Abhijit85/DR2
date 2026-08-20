# ERGO

**Evidential Rollback and Gap-triggered retrieval for diffusion language models.**

Reference implementation backing the ERGO paper (see `ergo-paper-blueprint.md` /
design doc). Two core contributions, wrapped in a diffusion-native ReAct harness:

- **C1 — Evidential rollback** (`ergo/rollback.py`): every committed token is a
  lease. After each evidence injection, a likelihood-ratio test
  `log p'(x) < log π(x) − δ` clears contradicted commits back to `[MASK]`,
  with `δ = log(1/α_roll)` — the threshold *is* a target false-rollback rate
  (bounded by `e^{−δ}` under the calibration idealization; blueprint Props 1–2).
- **C2 — Knowledge-gap trigger** (`ergo/trigger.py`): retrieval fires on spans
  with a *confident frame + uncertain details*, scored
  `G(S) = Rel(S,q)·conf(F)·u(D)` and gated by a **conformal rule** with
  distribution-free false-fire control at `α_fire` (Prop 3). The `μ+λσ`
  outlier rule remains as a calibration-free fallback.

Harness (credited to prior work, ablated not claimed): SPREAD relevance-guided
remasking (arXiv:2601.11342), SARDI-style lookahead retrieval loop
(arXiv:2606.06474), in-template ReAct canvas (cf. DLLM-Searcher,
arXiv:2602.07035), salience-fused ChromaDB retrieval, critique-driven evidence
gating, parallel-trajectory action voting.

## Layout

```
ergo/
  config.py          hyperparameters; contribution knobs are error rates
  canvas.py          [THOUGHT][ACTION][ACTION_INPUT][OBSERVATION][CRITIQUE][ANSWER]
  orchestrator.py    the ERGO loop (denoise → decide → retrieve → rollback → remask)
  rollback.py        C1
  trigger.py         C2 (+ ConformalCalibrator)
  salience.py        conf × Rel × IDF word extraction, trajectory pooling
  metrics.py         EM/F1, Copy Rate, RSD
  memory.py          working + episodic memory
  backbones/
    base.py          Snapshot / SPREAD selection / uncertainty scores
    mock.py          CPU mock dLLM (tests, dev)
    hf_adapter.py    LLaDABackbone + DreamBackbone (manual loop, shared policy)
  rag/
    chunker.py       2000-char word-aligned chunking + corpus IDF
    retriever.py     ChromaDB fused dual-embedding search + MMR; BM25 (SARDI parity)
    gating.py        per-chunk gates from critique verdicts + agreement
scripts/
  ingest_corpus.py   corpus → ChromaDB + idf.json
  run_query.py       single query (mock or GPU backbone)
  run_eval.py        benchmark runner (5 SARDI-parity datasets; CofCA/SynthWorlds TODO)
tests/               24 CPU tests — no GPU/torch needed
```

## Quickstart

```bash
pip install -r requirements.txt          # torch/transformers only needed on GPU box
python -m pytest tests/ -q               # 24 passed, CPU-only

# CPU smoke test of the full loop (mock backbone):
python scripts/run_query.py --backbone mock \
  --query "what is the capital city of France" \
  --mock-doc "The capital city of France is Paris and it hosts the Louvre museum." \
  --mock-fact "louvre museum=Paris is the capital"

# GPU (lab server):
python scripts/ingest_corpus.py --input corpus.jsonl --store ./stores/hotpotqa
python scripts/run_query.py --backbone dream --store ./stores/hotpotqa \
  --query "..." --device cuda
python scripts/run_eval.py --dataset hotpotqa --backbone dream \
  --store ./stores/hotpotqa --limit 200 --out results/hotpot_dream.jsonl
```

## Status / known TODOs for the experiment phase

1. **HF adapters are implemented but not yet run against real checkpoints**
   (this repo was scaffolded off-GPU). First GPU session: verify
   `LLaDABackbone` (mask id 126336, chat template) and `DreamBackbone`
   forward/decode on a toy prompt; check `_query_span` marker alignment
   against each tokenizer's chat template output.
2. **Conformal calibration harvest**: run dev traces, collect null-span scores
   (`ConformalCalibrator.add_null_scores`), persist per backbone+dataset.
3. **CofCA / SynthWorlds-RM**: wire SARDI's released counterfactual corpora
   into `run_eval.py` (marked TODO).
4. **Ablation flags**: each mechanism toggles via config (`use_conformal`,
   `alpha_roll→∞` ≈ no-rollback via `cap_fraction=0`, `n_parallel=1`, β=0);
   an explicit `--ablate` CLI on run_eval.py is worth adding before the sweep.
5. Tier-2 attention-bias gating (design doc §7.5) is not implemented — Tier-1
   eviction/demotion is, and ships first per the build order.
6. Efficiency: `hf_adapter` runs one forward per denoise step per trajectory;
   batch the N trajectories into one padded batch before the main runs.
