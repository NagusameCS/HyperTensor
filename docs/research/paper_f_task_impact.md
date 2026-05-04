# Paper F: GRC Task-Level Impact

Status: Data infrastructure complete. Awaiting benchmark dataset execution.
Target venue: arXiv cs.LG / cs.CL
Estimated completion: 2--4 weeks (1 EC2 session + analysis)

## Abstract (draft)

Paper A established that GRC attention compression preserves WikiText-2 perplexity
within +13.30% at throughput-preserving rank k=1536. But PPL is a proxy --- practitioners
care about task performance. This paper measures GRC-compressed models on MMLU
(57 subjects), GSM8K (math reasoning), HumanEval (code generation), and IFEval
(instruction following) at k ∈ {256, 512, 768, 1024, 1536, ∞} on
Llama-3.1-8B-Instruct Q4_K_M. We report the rank at which each benchmark drops
below 95% of baseline, establishing the "safe compression frontier" for deployment.

## Key Claims

1. MMLU is robust to attention compression: Predicted <2% drop at k≥1024.
   Rationale: multiple-choice knowledge retrieval depends more on FFN memory than
   attention routing.

2. GSM8K is sensitive: Predicted >5% drop at k<1536.
   Rationale: multi-step math reasoning requires precise attention routing across
   intermediate computation steps.

3. HumanEval is bimodal: Easy problems (pass@1 >80% at baseline) survive k=768;
   hard problems collapse below k=1024.

4. IFEval is the most sensitive: Instruction-following requires the model to
   attend to specific constraint tokens; compressed attention blurs these.

5. The "safe compression frontier" is k=1024 across all benchmarks --- the same
   rank that Paper A found as the cache-fit super-baseline.

## Data Needed

- MMLU test set (14,042 questions across 57 subjects)
- GSM8K test set (1,319 questions)
- HumanEval (164 problems)
- IFEval (541 prompts)

## Scripts Already Built

- `scripts/task_bench.py` --- full MMLU/GSM8K/HumanEval harness
- `scripts/multi_dataset_ppl.py` --- multi-dataset PPL evaluator

## Execution Plan

1. Download benchmark datasets -> `data/`
2. Run `task_bench.py` at k ∈ {256, 512, 768, 1024, 1536, ∞} on Llama-8B
3. Per-benchmark analysis: accuracy vs k, safe-compression frontier
4. Cross-benchmark correlation: does PPL predict task degradation?
5. Write-up: ~8 pages, 4 figures, 3 tables

## Key Figure Ideas

1. Fig 1: MMLU accuracy vs k, 57-subject heatmap (subjects  k)
2. Fig 2: GSM8K accuracy vs k, with solve-rate breakdown by steps
3. Fig 3: HumanEval pass@k vs compression rank, easy/hard split
4. Fig 4: Unified "safe compression frontier" --- all benchmarks overlaid
