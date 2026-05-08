# MedQA Error Analysis

This analysis treats review flags as a human-in-the-loop governance entry point. The system does not use LLM output directly as clinical decision support.

Blockchain logging records inference metadata, hashes, timestamps, and audit trajectories. It does not guarantee medical correctness and does not remove hallucination risk.

Observed failure categories include invalid answer formatting, wrong answers, low-confidence outputs, and unusually long generation latency. Hallucination and medical reasoning failures require clinician review, retrieval grounding, stronger backbones, and task-specific validation in future work.

Summary table:

```csv
model_name,num_questions,invalid_output_rate,wrong_answer_rate,review_flag_rate,review_precision_proxy,latency_failure_correlation,most_common_failure_type
Qwen/Qwen2.5-1.5B-Instruct,200,0.0,0.57,0.1,0.8,0.04157104824430028,wrong_answer
Qwen/Qwen2.5-3B-Instruct,200,0.0,0.5,0.1,0.55,0.07029964117670176,wrong_answer
Qwen/Qwen2.5-7B-Instruct,200,0.0,0.405,0.1,0.4,0.09234266320563943,wrong_answer
```
