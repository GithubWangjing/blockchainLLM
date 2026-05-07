chcp 65001 | Out-Null
$env:PYTHONIOENCODING = "utf-8"
$python = "F:\software\Anaconda\envs\blockchain\python.exe"
& $python -m experiments.run_medqa_backbone_comparison --quick --seed 7 --max-questions 50
& $python -m experiments.run_medqa_error_analysis --seed 7
& $python -m experiments.run_backbone_cost_scalability --seed 7
& $python -m experiments.generate_reviewer5_reports --seed 7
