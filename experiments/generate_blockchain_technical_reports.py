"""Generate blockchain technical evaluation report and v3 manifest."""

from __future__ import annotations

import argparse
import json

import pandas as pd

from experiments.reviewer_experiment_utils import FIGURES_DIR, PROJECT_ROOT, REPORTS_DIR, RESULTS_DIR, TABLES_DIR, ensure_output_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate blockchain technical reports.")
    parser.add_argument("--quick", action="store_true", help="Accepted for pipeline symmetry.")
    parser.add_argument("--seed", type=int, default=7, help="Recorded seed.")
    return parser.parse_args()


def rel(path):
    return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")


def write_mapping() -> None:
    rows = [
        {
            "reviewer5_comment": "Blockchain specification and access control are under-specified.",
            "required_revision": "Add contract-operation benchmark for grant/revoke/verify/log/emergency access.",
            "experiment_or_text_revision": "experiments/run_contract_access_control_benchmark.py",
            "output_files": "results/contract_access_control_benchmark.csv; figures/fig_contract_*",
            "manuscript_location": "Blockchain implementation and evaluation subsection",
            "interpretation": "Access-control operations add bounded devnet confirmation latency and create audit logs for state-changing operations.",
            "limitation_statement": "Current repository has no deployed Solidity access-control contract; latency uses real devnet tx benchmark with deterministic contract semantics.",
        },
        {
            "reviewer5_comment": "Tamper evidence is claimed but not evaluated.",
            "required_revision": "Evaluate hash-based audit verification against data/output/audit/model-update tampering.",
            "experiment_or_text_revision": "experiments/run_tamper_evidence_evaluation.py",
            "output_files": "results/tamper_evidence_evaluation.csv; figures/fig_tamper_*",
            "manuscript_location": "Auditability and security evaluation",
            "interpretation": "Hash registry detects modified off-chain records and missing audit records.",
            "limitation_statement": "Tamper evidence does not prove medical correctness and depends on key custody and honest registry writes.",
        },
        {
            "reviewer5_comment": "Consensus scalability and validator assumptions are unclear.",
            "required_revision": "Compare PoA-like, Raft-like, and PBFT-like consortium assumptions under validator/load/failure sweeps.",
            "experiment_or_text_revision": "experiments/run_consensus_scalability_simulation.py",
            "output_files": "results/consensus_scalability_simulation.csv; figures/fig_consensus_*",
            "manuscript_location": "Scalability evaluation and limitations",
            "interpretation": "Consensus choice changes finality latency, communication overhead, and failure tolerance.",
            "limitation_statement": "Consensus results are simulation, not a production consortium deployment.",
        },
        {
            "reviewer5_comment": "Cost and compliance details are missing.",
            "required_revision": "Clarify normalized gas/cost, off-chain storage, and governance/compliance limitations.",
            "experiment_or_text_revision": "reports/blockchain_technical_evaluation.md",
            "output_files": "tables/table_blockchain_reviewer5_mapping.csv",
            "manuscript_location": "Discussion and limitations",
            "interpretation": "On-chain state stores hashes/metadata rather than PHI-bearing records.",
            "limitation_statement": "GDPR/HIPAA, key management, and consortium governance require manuscript text and deployment policy.",
        },
    ]
    (TABLES_DIR / "table_blockchain_reviewer5_mapping.csv").write_text(pd.DataFrame(rows).to_csv(index=False), encoding="utf-8")


def write_report(seed: int) -> None:
    access = pd.read_csv(RESULTS_DIR / "contract_access_control_benchmark.csv")
    tamper = pd.read_csv(RESULTS_DIR / "tamper_evidence_evaluation.csv")
    consensus = pd.read_csv(RESULTS_DIR / "consensus_scalability_simulation.csv")
    text = f"""# Blockchain Technical Evaluation

Seed: `{seed}`.

## Evidence Types

- Access-control microbenchmark: `{access['benchmark_type'].iloc[0]}`. It reuses real Hardhat/devnet transaction latency from `prototypes/blockchain_local/results/` and applies deterministic contract-operation semantics because the current repository does not include a deployed Solidity access-control contract.
- Tamper-evidence evaluation: `{tamper['benchmark_type'].iloc[0]}`. It computes real SHA-256 hashes over workflow records and verifies modified/deleted/replaced records against an audit registry.
- Consensus scalability: `{consensus['benchmark_type'].iloc[0]}`. It is a simulation comparing PoA-like, Raft-like, and PBFT-like consortium assumptions.

## Security Interpretation

Blockchain guarantees tamper-evident auditability for logged hashes and metadata. It does not guarantee medical correctness, factuality, or clinical safety of LLM outputs.

Off-chain data are not stored on-chain. The modeled workflow logs hashes and metadata such as data hashes, inference hashes, model-update hashes, access-log hashes, timestamps, and operation identifiers.

## Access Control

The access-control benchmark covers `grantAccess`, `revokeAccess`, `verifyAccess`, `logInference`, `logModelUpdate`, `emergencyAccess`, and an unauthorized logging attempt. State-changing operations create audit logs; unauthorized operations are expected to be rejected.

## Tamper Evidence

Tamper attacks include modified off-chain data, modified inference output, deleted audit record, and replaced model update. Detection relies on mismatch between recomputed off-chain hashes and registered audit hashes.

## Consensus Scalability

PoA-like, Raft-like, and PBFT-like assumptions trade off latency, communication overhead, and fault/malicious-validator tolerance. These results should be reported as simulation unless replaced by a multi-node consortium deployment.

## Required Manuscript Text

- Key management and role assignment must be specified.
- GDPR/HIPAA compliance requires policy, access logging, retention, deletion, and data minimization discussion.
- Consortium governance must define validator admission/removal, emergency access policy, audit review, and incident response.
- Auditability supports accountability and tamper evidence, not automated clinical validation.

## Key Outputs

- `results/contract_access_control_benchmark.csv`
- `results/tamper_evidence_evaluation.csv`
- `results/consensus_scalability_simulation.csv`
- `tables/table_blockchain_reviewer5_mapping.csv`
"""
    (REPORTS_DIR / "blockchain_technical_evaluation.md").write_text(text, encoding="utf-8")

    summary = f"""# Experiment Summary V3

This v3 summary adds blockchain technical evaluation for Reviewer 5.

## New Blockchain Outputs

- Access-control benchmark: `results/contract_access_control_benchmark.csv`
- Tamper-evidence evaluation: `results/tamper_evidence_evaluation.csv`
- Consensus scalability simulation: `results/consensus_scalability_simulation.csv`
- Technical report: `reports/blockchain_technical_evaluation.md`

## Interpretation

The blockchain layer provides tamper-evident auditability, access-control logging, and accountable workflow metadata. It does not improve or certify medical answer correctness.

Formal evidence status:

- Real devnet-derived/hybrid: access-control operation latency.
- Real hash verification: tamper-evidence audit checks.
- Simulation: consensus/validator scalability.

Compliance and governance topics, including key management and GDPR/HIPAA, require manuscript text discussion and cannot be fully solved by microbenchmarks.
"""
    (REPORTS_DIR / "experiment_summary_v3.md").write_text(summary, encoding="utf-8")


def write_manifest() -> None:
    manifest = {
        "results": sorted(rel(p) for p in RESULTS_DIR.glob("*.csv")),
        "tables": sorted(rel(p) for p in TABLES_DIR.glob("*.csv")),
        "figures": sorted(rel(p) for p in FIGURES_DIR.glob("*.*") if p.suffix.lower() in [".png", ".pdf", ".svg"]),
        "reports": sorted(rel(p) for p in REPORTS_DIR.glob("*.md")),
    }
    (RESULTS_DIR / "experiment_manifest_v3.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dirs()
    write_mapping()
    write_report(args.seed)
    write_manifest()
    print("Wrote blockchain technical reports and manifest v3")


if __name__ == "__main__":
    main()

