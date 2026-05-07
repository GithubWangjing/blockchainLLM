# Blockchain Technical Evaluation

Seed: `7`.

## Evidence Types

- Access-control microbenchmark: `hybrid_real_devnet_latency_plus_contract_semantics`. It reuses real Hardhat/devnet transaction latency from `prototypes/blockchain_local/results/` and applies deterministic contract-operation semantics because the current repository does not include a deployed Solidity access-control contract.
- Tamper-evidence evaluation: `real_hash_verification_on_workflow_records`. It computes real SHA-256 hashes over workflow records and verifies modified/deleted/replaced records against an audit registry.
- Consensus scalability: `consensus_scalability_simulation`. It is a simulation comparing PoA-like, Raft-like, and PBFT-like consortium assumptions.

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
