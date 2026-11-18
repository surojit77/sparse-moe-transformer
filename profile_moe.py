"""
GPU profiling: latency, throughput, peak memory and analytic FLOPs for
Dense vs Top-1 vs Top-2 MoE, for each dispatch implementation, across batch sizes,
sequence lengths and expert counts. Also a torch.profiler breakdown of where MoE
time goes (router / dispatch / expert compute / combine).

  python profile_moe.py                  # full sweep -> results/bench.csv, results/breakdown.json
  python profile_moe.py --quick          # small sweep
"""
import argparse
import csv
import itertools
import json
import os
import statistics
import time

import torch
from torch.profiler import ProfilerActivity, profile

from moe.data import amp_dtype_for
from moe.model import DenseFFN, ModelConfig, MoEFFN, Transformer


CF = 1.25  # capacity factor used for BOTH train and eval while benchmarking


def build(ffn, top_k, E, d_ff, dispatch, max_seq, device, dtype, vocab=50257, layer_only=False,
          fast_router=True):
    cfg = ModelConfig(vocab_size=vocab, seq_len=max_seq, ffn_type=ffn, top_k=top_k, n_experts=E,
                      d_ff=d_ff, dispatch=dispatch, dropout=0.0,
                      capacity_factor=CF, eval_capacity_factor=CF, fast_router=fast_router)
    if layer_only:                     # just one FFN block: isolates MoE overhead
        m = (MoEFFN(cfg) if ffn == "moe" else DenseFFN(cfg)).to(device=device, dtype=dtype)
        m.cfg = cfg
        return m
    return Transformer(cfg).to(device=device, dtype=dtype)


def timeit(fn, device, warmup, iters):
    for _ in range(warmup):
        fn()
    times = []
    if device == "cuda":
        torch.cuda.synchronize()
        for _ in range(iters):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record(); fn(); e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))
    else:
        for _ in range(iters):
            t = time.perf_counter(); fn(); times.append((time.perf_counter() - t) * 1e3)
    return statistics.median(times)


def bench_one(m, mode, B, T, device, warmup, iters):
    x = torch.randint(0, m.cfg.vocab_size, (B, T), device=device)
    if mode == "layer":                # single FFN layer forward on [B, T, d] activations
        m.eval()
        h = torch.randn(B, T, m.cfg.d_model, device=device, dtype=next(m.parameters()).dtype)
        def fn():
            with torch.no_grad():
                m(h)
    elif mode == "inference":
        m.eval()
        def fn():
            with torch.no_grad():
                m(x)
    else:  # train step: fwd + bwd
        m.train()
        def fn():
            _, lm, aux, _ = m(x, x)
            (lm + aux).backward()
            m.zero_grad(set_to_none=True)
    if device == "cuda":
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    ms = timeit(fn, device, warmup, iters)
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else float("nan")
    return ms, mem


def breakdown(m, B, T, device):
    """Fraction of device time spent in each labelled MoE stage."""
    m.eval()
    x = torch.randint(0, m.cfg.vocab_size, (B, T), device=device)
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if device == "cuda" else [])
    with torch.no_grad():
        for _ in range(5):
            m(x)
        if device == "cuda":
            torch.cuda.synchronize()
        with profile(activities=acts) as prof:
            for _ in range(10):
                m(x)
            if device == "cuda":
                torch.cuda.synchronize()

    def dev_t(evt, self_only):
        if device == "cuda":
            names = ["self_device_time_total", "self_cuda_time_total"] if self_only \
                else ["device_time_total", "cuda_time_total"]
        else:
            names = ["self_cpu_time_total"] if self_only else ["cpu_time_total"]
        for n in names:
            if hasattr(evt, n):
                return getattr(evt, n)
        return 0.0

    ev = prof.key_averages()
    total = sum(dev_t(e, True) for e in ev)
    out = {}
    for e in ev:
        if e.key.startswith("moe/"):
            out[e.key] = round(100 * dev_t(e, False) / max(total, 1e-9), 2)
    out["moe_total_pct"] = round(sum(out.values()), 2)
    return out, prof


def router_bench(a, dev, dtype):
    """Original router (one-hot positions, losses always computed) vs fast router."""
    variants = [("top1", 1, 1024), ("top2_matched", 2, 512)]
    rows = []
    for mode in ("layer", "inference", "train"):
        dense = {}
        for B in a.router_batches:
            m = build("dense", 1, 8, 1024, "batched", 128, dev, dtype, layer_only=(mode == "layer"))
            dense[B], _ = bench_one(m, mode, B, 128, dev, a.warmup, a.iters); del m
        for (label, k, dff), E, B in itertools.product(variants, (8, 16), a.router_batches):
            r = {"mode": mode, "variant": label, "n_experts": E, "batch": B, "seq": 128,
                 "dense_ms": round(dense[B], 3)}
            for fast in (False, True):
                m = build("moe", k, E, dff, "batched", 128, dev, dtype,
                          layer_only=(mode == "layer"), fast_router=fast)
                ms, _ = bench_one(m, mode, B, 128, dev, a.warmup, a.iters); del m
                r["fast_ms" if fast else "orig_ms"] = round(ms, 3)
            r["speedup"] = round(r["orig_ms"] / r["fast_ms"], 3)
            r["slowdown_vs_dense_orig"] = round(r["orig_ms"] / r["dense_ms"], 3)
            r["slowdown_vs_dense_fast"] = round(r["fast_ms"] / r["dense_ms"], 3)
            rows.append(r); print(r, flush=True)
    with open(os.path.join(a.out_dir, "router_bench.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    bd = {}
    for label, k, dff in variants:
        for fast in (False, True):
            m = build("moe", k, 8, dff, "batched", 128, dev, dtype, fast_router=fast)
            bd[f"{label}/batched/{'fast' if fast else 'orig'}_router"], _ = breakdown(m, 32, 128, dev)
            del m
    with open(os.path.join(a.out_dir, "router_breakdown.json"), "w") as f:
        json.dump(bd, f, indent=2)
    print(json.dumps(bd, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--quick", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="auto", choices=["auto", "fp32", "fp16", "bf16"])
    p.add_argument("--out_dir", default="results")
    p.add_argument("--iters", type=int, default=30)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--capacity_factor", type=float, default=1.25)
    p.add_argument("--router_only", action="store_true",
                   help="only compare original vs fast router (batched dispatch)")
    p.add_argument("--router_batches", type=lambda v: [int(x) for x in v.split(",")], default=[1, 32, 64])
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    global CF
    CF = a.capacity_factor
    dev = a.device
    if a.dtype == "auto":
        dtype = amp_dtype_for(dev) or torch.float32
    else:
        dtype = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[a.dtype]
    print(f"device={dev} dtype={dtype}" +
          (f" gpu={torch.cuda.get_device_name()}" if dev == "cuda" else ""))

    if a.router_only:
        router_bench(a, dev, dtype)
        return

    # (label, ffn, top_k, d_ff) -- d_ff chosen so that "matched" variants use the same
    # active FFN FLOPs per token as the dense baseline (d_ff=1024).
    variants = [
        ("dense",         "dense", 1, 1024),
        ("top1",          "moe",   1, 1024),   # same active FLOPs as dense
        ("top2_matched",  "moe",   2, 512),    # same active FLOPs as dense
        ("top2_full",     "moe",   2, 1024),   # 2x active FFN FLOPs
    ]
    dispatches = ["naive", "sorted", "batched"]
    if a.quick:
        batch_sizes, seq_lens, experts, modes = [8, 32], [128], [8], ["layer", "inference"]
    else:
        batch_sizes, seq_lens, experts, modes = [1, 8, 32, 64], [128, 512], [2, 4, 8, 16], \
            ["layer", "inference", "train"]
    max_seq = max(seq_lens)

    rows, dense_ref = [], {}
    for mode, (label, ffn, k, dff), E, dispatch, B, T in itertools.product(
            modes, variants, experts, dispatches, batch_sizes, seq_lens):
        if ffn == "dense" and (dispatch != "naive" or E != experts[0]):
            continue                                  # dense has no experts/dispatch
        if ffn == "moe" and k > E:
            continue
        m = build(ffn, k, E, dff, dispatch, max_seq, dev, dtype, layer_only=(mode == "layer"))
        try:
            ms, mem = bench_one(m, mode, B, T, dev, a.warmup, a.iters)
        except torch.OutOfMemoryError:
            print("OOM", label, E, dispatch, B, T); del m; torch.cuda.empty_cache(); continue
        pc = m.param_counts() if mode != "layer" else {"active": float("nan"), "total": float("nan")}
        row = {
            "mode": mode, "variant": label, "n_experts": E if ffn == "moe" else 0,
            "top_k": k if ffn == "moe" else 0, "d_ff": dff,
            "dispatch": dispatch if ffn == "moe" else "-", "batch": B, "seq": T,
            "latency_ms": round(ms, 3), "tokens_per_s": round(B * T / ms * 1e3),
            "peak_mem_mb": round(mem, 1),
            "gflops_fwd": round(m.flops_per_token(T) * B * T / 1e9, 3) if mode != "layer" else "",
            "active_params_m": round(pc["active"] / 1e6, 2),
            "total_params_m": round(pc["total"] / 1e6, 2),
        }
        if ffn == "dense":
            dense_ref[(mode, B, T)] = ms
        row["slowdown_vs_dense"] = round(ms / dense_ref[(mode, B, T)], 3) \
            if (mode, B, T) in dense_ref else ""
        rows.append(row)
        print(row, flush=True)
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()

    path = os.path.join(a.out_dir, "bench.csv")
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print("wrote", path)

    # Where does MoE time go?  (E=8, B=32, T=128)
    bd = {}
    for dispatch in dispatches:
        for label, ffn, k, dff in variants[1:3]:
            m = build(ffn, k, 8, dff, dispatch, max_seq, dev, dtype)
            res, prof = breakdown(m, 32, 128, dev)
            bd[f"{label}/{dispatch}"] = res
            if dispatch == "batched" and label == "top2_matched" and dev == "cuda":
                prof.export_chrome_trace(os.path.join(a.out_dir, "trace_top2_batched.json"))
            del m
    with open(os.path.join(a.out_dir, "breakdown.json"), "w") as f:
        json.dump(bd, f, indent=2)
    print(json.dumps(bd, indent=2))


if __name__ == "__main__":
    main()
