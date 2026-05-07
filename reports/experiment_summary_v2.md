# Experiment Summary V2

Seed used for Reviewer 5 quick extension: `7`.

## Reviewer 5 Focus

This extension adds a MedQA backbone comparison, error/invalid-output analysis, and normalized backbone cost scalability analysis. It reinforces the interpretation that MedChainLLM does not aim to improve intrinsic medical reasoning ability. Absolute MedQA accuracy is backbone-dependent; blockchain integration preserves model predictions while adding auditability, governance, and tamper-evident workflow logging.

## Real, Hybrid, Simulation, and Mock Status

- Real backbone evaluations included in formal manuscript outputs: `Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct`.
- Not evaluated / failed models: `none`.
- Hybrid analyses: blockchain workflow rows for real 0.5B predictions, because logging overhead is added to fixed real predictions.
- Exclusion rule: Only real backbone evaluations are included in the manuscript figures and tables. Mock or dry-run outputs, if any, were used solely for pipeline testing and excluded from scientific analysis.
- Simulation/hybrid analyses: normalized cost, review flag proxies, prior DP/FL simulations, and blockchain logging overhead estimates.
- No result should be described as clinical validation.

## Backbone Accuracy and Latency

```csv
model_name,parameter_size,num_questions,accuracy,mean_latency_ms,benchmark_type
Qwen/Qwen2.5-0.5B,0.5B,50,0.2,2169.7676579956897,real
Qwen/Qwen2.5-1.5B-Instruct,1.5B,50,0.36,8104.340814000461,real
Qwen/Qwen2.5-3B-Instruct,3B,50,0.44,94658.07317200815,real

```

Blockchain overhead percentages by backbone workflow:

```text
[{'parameter_size': '0.5B', 'blockchain_overhead_percent': 39.94930931661606}, {'parameter_size': '1.5B', 'blockchain_overhead_percent': 11.170479867746169}, {'parameter_size': '3B', 'blockchain_overhead_percent': 0.8351014330592811}]
```

## Error and Governance Summary

Invalid output, wrong answer, and review flag rates are summarized in `results/medqa_error_analysis.csv`. Review flags are a human-in-the-loop governance entry point, not automated clinical adjudication. Blockchain records inference process and audit trail; it does not guarantee medical correctness.

## Cost Scalability

`results/backbone_cost_scalability.csv` reports normalized inference cost, blockchain logging cost, cost per 1000 queries, audit coverage, latency, and accuracy. Normalized cost is used to compare relative deployment overhead across configurations.

## Updated Figure/Table Recommendations

- Main Figure 2: Sharding scalability and blockchain performance: `fig_sharding_throughput_vs_load`, `fig_sharding_latency_vs_validators`, `fig_dynamic_shard_adaptation`.
- Main Figure 3: MedQA backbone and workflow evaluation: `fig_backbone_accuracy_latency_tradeoff`, `fig_backbone_latency_breakdown`, `fig_backbone_auditability_overhead`.
- Main Figure 4: Privacy, FL, and governance tradeoffs: `fig_privacy_accuracy_vs_epsilon`, `fig_fl_accuracy_over_rounds`, `fig_medqa_review_flag_by_failure_type` or `fig_medqa_invalid_output_by_model`.
- Main Table 2: Baseline, ablation, and backbone comparison.
- Main Table 3: Cost, auditability, and deployment overhead.
- Supplementary: full MedQA detail, error analysis, cost scalability, FL logs, full DP parameters, and mock fallback notes.

## Remaining Limitations for Manuscript Text

- Qwen2.5-1.5B/3B/7B are listed as not evaluated when download, permissions, memory, or runtime prevented completion; they are excluded from formal figures/tables until real predictions are available.
- Clinical safety cannot be inferred from MedQA accuracy alone.
- Hallucination mitigation requires retrieval grounding, clinician review, stronger backbones, and prospective validation.
- Blockchain adds auditability and tamper evidence but depends on validator governance, key management, and consortium trust assumptions.
