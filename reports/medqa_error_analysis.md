# MedQA Error Analysis

This analysis treats review flags as a human-in-the-loop governance entry point. The system does not use LLM output directly as clinical decision support.

Blockchain logging records inference metadata, hashes, timestamps, and audit trajectories. It does not guarantee medical correctness and does not remove hallucination risk.

Observed failure categories include invalid answer formatting, wrong answers, low-confidence outputs, and unusually long generation latency. Hallucination and medical reasoning failures require clinician review, retrieval grounding, stronger backbones, and task-specific validation in future work.

Summary table:

```csv
model_name,num_questions,invalid_output_rate,wrong_answer_rate,review_flag_rate,review_precision_proxy,latency_failure_correlation,most_common_failure_type
Qwen/Qwen2.5-1.5B-Instruct,50,0.0,0.64,0.1,0.6,0.015665816777376426,wrong_answer
```
