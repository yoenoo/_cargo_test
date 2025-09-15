set -e

# python3 inference.py "Qwen/Qwen3-Coder-30B-A3B-Instruct" cargo_test_passed_eval.parquet qwen3_coder_30b_a3b_instruct_results.parquet --deterministic
# python3 eval.py qwen3_coder_30b_a3b_instruct_results.parquet qwen3_coder_30b_a3b_instruct_results_eval.parquet

python3 inference.py "Qwen/Qwen3-4B" cargo_test_passed_eval.parquet qwen3_4b_results.parquet --deterministic --max-tokens 8192
python3 eval.py qwen3_4b_results.parquet qwen3_4b_results_eval.parquet
