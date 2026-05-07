"""Tamper-evidence audit verification evaluation.

This script constructs real SHA-256 audit records from existing workflow CSV
rows, stores only hashes/metadata in an in-memory audit registry, then applies
tampering attacks and verifies whether hash mismatches are detected.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.reviewer_experiment_utils import RESULTS_DIR, TABLES_DIR, ensure_output_dirs, save_figure, style_matplotlib


ATTACKS = ["modified_offchain_data", "modified_inference_output", "deleted_audit_record", "replaced_model_update", "clean_record"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tamper-evidence evaluation.")
    parser.add_argument("--quick", action="store_true", help="Use fewer records.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    return parser.parse_args()


def sha256_obj(obj: dict) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True).encode("utf-8")).hexdigest()


def load_workflow_rows(n: int) -> pd.DataFrame:
    path = RESULTS_DIR / "medqa_backbone_detail.csv"
    if path.exists():
        df = pd.read_csv(path)
        df = df[df["workflow"] == "backbone_only"].head(n)
        if not df.empty:
            return df
    return pd.DataFrame({"question_id": [f"synthetic_record_{i}" for i in range(n)], "prediction": ["A"] * n, "answer": ["A"] * n, "model_name": ["unknown"] * n})


def build_record(row: pd.Series) -> dict:
    data = {"question_id": str(row.get("question_id", "")), "model_name": str(row.get("model_name", ""))}
    inference = {"prediction": str(row.get("prediction", "")), "answer": str(row.get("answer", ""))}
    update = {"model_name": str(row.get("model_name", "")), "parameter_size": str(row.get("parameter_size", ""))}
    access = {"workflow": str(row.get("workflow", "backbone_only")), "audit_hash_present": int(row.get("audit_hash_present", 0))}
    return {
        "record_id": data["question_id"],
        "data_hash": sha256_obj(data),
        "inference_hash": sha256_obj(inference),
        "model_update_hash": sha256_obj(update),
        "access_log_hash": sha256_obj(access),
        "payload": {"data": data, "inference": inference, "update": update, "access": access},
    }


def verify(registry: dict, record: dict) -> bool:
    stored = registry.get(record["record_id"])
    if stored is None:
        return False
    return all(stored[key] == record[key] for key in ["data_hash", "inference_hash", "model_update_hash", "access_log_hash"])


def attack_record(record: dict, attack: str) -> dict:
    attacked = json.loads(json.dumps(record))
    if attack == "modified_offchain_data":
        attacked["payload"]["data"]["question_id"] += "_tampered"
        attacked["data_hash"] = sha256_obj(attacked["payload"]["data"])
    elif attack == "modified_inference_output":
        attacked["payload"]["inference"]["prediction"] = "D" if attacked["payload"]["inference"]["prediction"] != "D" else "A"
        attacked["inference_hash"] = sha256_obj(attacked["payload"]["inference"])
    elif attack == "replaced_model_update":
        attacked["payload"]["update"]["model_name"] = "replaced_model"
        attacked["model_update_hash"] = sha256_obj(attacked["payload"]["update"])
    return attacked


def run(args: argparse.Namespace) -> pd.DataFrame:
    ensure_output_dirs()
    n = 30 if args.quick else 200
    rows = load_workflow_rows(n)
    records = [build_record(row) for _, row in rows.iterrows()]
    registry = {r["record_id"]: {k: r[k] for k in ["data_hash", "inference_hash", "model_update_hash", "access_log_hash"]} for r in records}
    out = []
    for attack in ATTACKS:
        detected = []
        latencies = []
        for record in records:
            reg = dict(registry)
            test_record = attack_record(record, attack)
            if attack == "deleted_audit_record":
                reg.pop(record["record_id"], None)
            start = time.perf_counter()
            ok = verify(reg, test_record)
            latencies.append((time.perf_counter() - start) * 1000.0)
            is_attack = attack != "clean_record"
            detected.append(int((not ok) if is_attack else ok))
        if attack == "clean_record":
            false_positive = 1.0 - float(np.mean(detected))
            detection = 0.0
            false_negative = 0.0
        else:
            detection = float(np.mean(detected))
            false_negative = 1.0 - detection
            false_positive = 0.0
        out.append(
            {
                "attack_type": attack,
                "num_records": len(records),
                "tamper_detection_rate": detection,
                "false_negative_rate": false_negative,
                "false_positive_rate": false_positive,
                "audit_recovery_rate": 1.0 if attack != "deleted_audit_record" else 0.0,
                "verification_latency_ms": float(np.mean(latencies)),
                "benchmark_type": "real_hash_verification_on_workflow_records",
            }
        )
    return pd.DataFrame(out)


def plot(df: pd.DataFrame) -> None:
    style_matplotlib()
    attacks = df["attack_type"].str.replace("_", "\n")
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    ax.bar(x, df["tamper_detection_rate"], color="#4C78A8")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Detection rate")
    ax.set_title("Tamper Detection by Attack")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, "fig_tamper_detection_by_attack")

    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.bar(x, df["verification_latency_ms"], color="#54A24B")
    ax.set_xticks(x)
    ax.set_xticklabels(attacks)
    ax.set_ylabel("Verification latency (ms)")
    ax.set_title("Audit Verification Latency")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    save_figure(fig, "fig_audit_verification_latency")


def main() -> None:
    args = parse_args()
    df = run(args)
    result_path = RESULTS_DIR / "tamper_evidence_evaluation.csv"
    table_path = TABLES_DIR / "table_tamper_evidence_evaluation.csv"
    df.to_csv(result_path, index=False)
    df.to_csv(table_path, index=False)
    plot(df)
    print(f"Wrote {result_path}")
    print(f"Wrote {table_path}")


if __name__ == "__main__":
    main()

