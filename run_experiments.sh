#!/usr/bin/env bash
# Full experiment suite.  SEEDS="0" STEPS=2000 bash run_experiments.sh  for a fast pass.
set -e
SEEDS=${SEEDS:-"0 1 2"}
STEPS=${STEPS:-4000}
COMMON="--steps $STEPS --batch_size 32 --seq_len 128"

for s in $SEEDS; do
  # Dense baseline
  python train.py --name dense_s$s --ffn dense --d_ff 1024 --seed $s $COMMON

  for E in 2 4 8 16; do
    # Top-1: each expert = dense FFN size  -> same active FLOPs/token as dense
    python train.py --name top1_e${E}_s$s --ffn moe --top_k 1 --n_experts $E --d_ff 1024 --seed $s $COMMON
    # Top-2 compute-matched: half-size experts -> same active FLOPs/token as dense
    python train.py --name top2m_e${E}_s$s --ffn moe --top_k 2 --n_experts $E --d_ff 512 --seed $s $COMMON
  done

  # Top-2 with full-size experts (2x active FFN compute) -- shows what an unmatched comparison looks like
  python train.py --name top2f_e8_s$s --ffn moe --top_k 2 --n_experts 8 --d_ff 1024 --seed $s $COMMON
done

python profile_moe.py
python analyze.py
