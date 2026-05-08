# Backbone Extension Plan

## Current MedQA Status

No backbone completed in this run.

## Evaluation Policy

Only real backbone evaluations are included in the manuscript figures and tables. Mock or dry-run outputs, if any, were used solely for pipeline testing and excluded from scientific analysis.

Medical QA accuracy must come from real model predictions on real MedQA samples. Blockchain logging overhead may use the existing real devnet benchmark as a hybrid workflow estimate, but it must not change correctness.

## Subset Protocol

- Requested maximum questions: `50`.
- Sampling seed: `7`.
- Sampling method: seeded sampling without replacement when more real MedQA questions are available than requested.

## Model Status

```csv
model_name,parameter_size,model_source,evaluation_status,failure_reason,benchmark_type
Qwen/Qwen2.5-0.5B,0.5B,local_existing_benchmark,completed,,real_status
Qwen/Qwen2.5-3B-Instruct,3B,local_dir,completed,,real_status
Qwen/Qwen2.5-1.5B-Instruct,1.5B,local_dir,completed,,real_status

```

## Relation to Prior Experiments

This extension reuses prior no-blockchain/blockchain baseline and cost/auditability outputs, but the backbone comparison itself includes only completed real model evaluations. Failed or unavailable models are marked not evaluated in the reports and excluded from formal figures.
