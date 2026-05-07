chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$python = "F:\software\Anaconda\envs\blockchain\python.exe"
if (!(Test-Path $python)) {
  $python = "python"
}
& $python -m experiments.run_contract_access_control_benchmark --quick --seed 7
& $python -m experiments.run_tamper_evidence_evaluation --quick --seed 7
& $python -m experiments.run_consensus_scalability_simulation --quick --seed 7
& $python -m experiments.generate_blockchain_technical_reports --quick --seed 7
