# Backbone Extension Plan

## Current MedQA Status

- Completed real backbones: `Qwen/Qwen2.5-0.5B, Qwen/Qwen2.5-1.5B-Instruct, Qwen/Qwen2.5-3B-Instruct`.
- Evaluated samples per backbone: `50`.
- Sampling seed: `7`.
- Sampling method: seeded subset sampling, with 0.5B reused from the existing real benchmark CSV and 1.5B/3B evaluated from the local MedQA JSON subset.

## Evaluation Policy

Only real backbone evaluations are included in the manuscript figures and tables. Mock or dry-run outputs, if any, were used solely for pipeline testing and excluded from scientific analysis.

Medical QA accuracy must come from real model predictions on real MedQA samples. Blockchain logging overhead may use the existing real devnet benchmark as a hybrid workflow estimate, but it must not change correctness.

## Completed Results

```csv
model_name,parameter_size,num_questions,accuracy,mean_latency_ms,model_source,benchmark_type
Qwen/Qwen2.5-0.5B,0.5B,50,0.2,2169.7676579956897,local_existing_benchmark,real
Qwen/Qwen2.5-1.5B-Instruct,1.5B,50,0.36,8104.340814000461,local_dir,real
Qwen/Qwen2.5-3B-Instruct,3B,50,0.44,94658.07317200815,local_dir,real

```

## Model Status

```csv
model_name,parameter_size,model_source,evaluation_status,failure_reason,benchmark_type
Qwen/Qwen2.5-0.5B,0.5B,local_existing_benchmark,completed,,real_status
Qwen/Qwen2.5-1.5B-Instruct,1.5B,local_dir,completed,,real_status
Qwen/Qwen2.5-3B-Instruct,3B,local_dir,completed,,real_status

```

## Relation to Prior Experiments

This extension reuses prior no-blockchain/blockchain baseline and cost/auditability outputs. The backbone comparison itself includes completed real model evaluations only; failed or unavailable models are excluded from formal figures.
