set -e

python3 inference.py "Qwen/Qwen3-4B" cargo_test_passed_eval.parquet qwen3_4b_lora_results.parquet --deterministic --max-tokens 8192 --lora-adapter-path /exploration_hacking/_cargo_test/qwen3-4b-checkpoints/checkpoint-1200
python3 eval.py qwen3_4b_lora_results.parquet qwen3_4b_lora_results_eval.parquet