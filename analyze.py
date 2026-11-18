"""Turn results/*.json + results/bench.csv into results/summary.md and plots."""
import csv
import glob
import json
import os
import statistics
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = "results"


def label(a):
    if a["ffn"] == "dense":
        return f"Dense (d_ff={a['d_ff']})"
    return f"Top-{a['top_k']} MoE, E={a['n_experts']}, d_ff={a['d_ff']}"


def ms(v):
    if len(v) == 1:
        return f"{v[0]:.2f}"
    return f"{statistics.mean(v):.2f} ± {statistics.stdev(v):.2f}"


def main():
    lines = ["# Results\n"]
    runs = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/*.json"))
            if not p.endswith(("breakdown.json",)) and "trace" not in p]
    runs = [r for r in runs if "test_ppl" in r]

    # ---------- quality table ----------
    if runs:
        groups = defaultdict(list)
        for r in runs:
            a = r["args"]
            groups[(a["ffn"], a["top_k"] if a["ffn"] == "moe" else 0,
                    a["n_experts"] if a["ffn"] == "moe" else 0, a["d_ff"])].append(r)
        lines += ["## Language modelling (WikiText-2 test perplexity, lower is better)\n",
                  "| Model | Seeds | Test PPL | Total params | Active params | Fwd GFLOPs/token | Train tok/s | Worst-layer max expert load |",
                  "|---|---|---|---|---|---|---|---|"]
        dense_ppl = None
        for key in sorted(groups):
            rs = groups[key]
            ppl = [r["test_ppl"] for r in rs]
            if key[0] == "dense":
                dense_ppl = statistics.mean(ppl)
            p = rs[0]["params"]
            load = rs[0]["expert_load"]
            maxload = max((max(l) for l in load), default=None)
            lines.append(
                f"| {label(rs[0]['args'])} | {len(rs)} | {ms(ppl)} | {p['total']/1e6:.1f}M | "
                f"{p['active']/1e6:.1f}M | {rs[0]['flops_per_token']/1e9:.4f} | "
                f"{statistics.mean(r['train_tokens_per_s'] for r in rs):,.0f} | "
                f"{'-' if maxload is None else f'{maxload*100:.0f}%'} |")
        lines.append("\n_Max expert load: share of routed tokens taken by the busiest expert "
                     "(1/E is perfectly balanced)._\n")
        if dense_ppl:
            lines.append(f"Dense baseline mean test PPL: {dense_ppl:.2f}\n")

        # PPL vs number of experts
        fig, ax = plt.subplots(figsize=(6, 4))
        for (ffn, k, dff), name in {("moe", 1, 1024): "Top-1 (d_ff=1024)",
                                    ("moe", 2, 512): "Top-2 matched (d_ff=512)",
                                    ("moe", 2, 1024): "Top-2 full (d_ff=1024)"}.items():
            pts = sorted((key[2], statistics.mean(r["test_ppl"] for r in rs),
                          statistics.stdev([r["test_ppl"] for r in rs]) if len(rs) > 1 else 0.0)
                         for key, rs in groups.items() if key[0] == ffn and key[1] == k and key[3] == dff)
            if pts:
                ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                            marker="o", capsize=3, label=name)
        if dense_ppl:
            ax.axhline(dense_ppl, ls="--", c="gray", label="Dense")
            dk = [key for key in groups if key[0] == "dense"]
            if dk and len(groups[dk[0]]) > 1:
                sd = statistics.stdev([r["test_ppl"] for r in groups[dk[0]]])
                ax.axhspan(dense_ppl - sd, dense_ppl + sd, color="gray", alpha=0.15)
        ax.set_xscale("log", base=2); ax.set_xlabel("number of experts"); ax.set_ylabel("test PPL (mean ± std over seeds)")
        ax.set_title("WikiText-2 perplexity vs experts"); ax.legend(); fig.tight_layout()
        fig.savefig(f"{R}/ppl_vs_experts.png", dpi=150)
        lines.append("![ppl](ppl_vs_experts.png)\n")

        # Expert utilisation heatmap for the largest-E top-1 run
        cand = [r for r in runs if r["args"]["ffn"] == "moe"]
        if cand:
            r = max(cand, key=lambda r: r["args"]["n_experts"])
            fig, ax = plt.subplots(figsize=(6, 2.5))
            im = ax.imshow(r["expert_load"], aspect="auto", cmap="viridis")
            ax.set_xlabel("expert"); ax.set_ylabel("layer"); ax.set_title(f"Expert load: {r['name']}")
            fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{R}/expert_load.png", dpi=150)
            lines.append("![load](expert_load.png)\n")

    # ---------- latency ----------
    if os.path.exists(f"{R}/bench.csv"):
        rows = list(csv.DictReader(open(f"{R}/bench.csv")))
        lines += ["## Inference latency, E=8, batch 32, seq 128\n",
                  "| Variant | Dispatch | Latency (ms) | Tokens/s | Peak mem (MB) | Slowdown vs dense |",
                  "|---|---|---|---|---|---|"]
        sel = [r for r in rows if r["mode"] == "inference" and r["batch"] == "32" and r["seq"] == "128"
               and r["n_experts"] in ("0", "8")]
        for r in sel:
            lines.append(f"| {r['variant']} | {r['dispatch']} | {r['latency_ms']} | {r['tokens_per_s']} | "
                         f"{r['peak_mem_mb']} | {r['slowdown_vs_dense']} |")
        # naive vs batched speed-up
        lines.append("\n### Speed-up of optimised dispatch over the naive per-expert loop\n"
                     "(layer = one isolated FFN layer; inference = full model forward)\n")
        lines += ["| Mode | Variant | E | Batch | Seq | Naive ms | Sorted ms | Batched ms | Best speed-up |",
                  "|---|---|---|---|---|---|---|---|---|"]
        idx = {(r["mode"], r["variant"], r["n_experts"], r["batch"], r["seq"], r["dispatch"]): r for r in rows}
        for (mode, v, E, B, T, d), r in idx.items():
            if mode in ("layer", "inference") and d == "naive" and v != "dense":
                b = idx.get((mode, v, E, B, T, "batched"))
                so = idx.get((mode, v, E, B, T, "sorted"))
                if b and so:
                    best = min(float(b["latency_ms"]), float(so["latency_ms"]))
                    lines.append(f"| {mode} | {v} | {E} | {B} | {T} | {r['latency_ms']} | {so['latency_ms']} | "
                                 f"{b['latency_ms']} | {float(r['latency_ms'])/best:.2f}x |")

        # slowdown vs dense as experts grow (full-model inference, batch 32, seq 128)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
        for ax, v in zip(axes, ("top1", "top2_matched")):
            for d in ("naive", "sorted", "batched"):
                pts = sorted((int(r["n_experts"]), float(r["slowdown_vs_dense"])) for r in rows
                             if r["mode"] == "inference" and r["variant"] == v and r["dispatch"] == d
                             and r["batch"] == "32" and r["seq"] == "128" and r["slowdown_vs_dense"])
                if pts:
                    ax.plot([p[0] for p in pts], [p[1] for p in pts], marker="o", label=d)
            ax.axhline(1.0, ls="--", c="gray", label="dense")
            ax.set_xscale("log", base=2); ax.set_xlabel("number of experts"); ax.set_title(v)
        axes[0].set_ylabel("latency / dense latency"); axes[0].legend()
        fig.suptitle("Full-model inference slowdown vs dense (batch 32, seq 128)")
        fig.tight_layout(); fig.savefig(f"{R}/slowdown_vs_experts.png", dpi=150)
        lines.append("![slowdown](slowdown_vs_experts.png)\n")

        # bar chart
        fig, ax = plt.subplots(figsize=(7, 4))
        names = [f"{r['variant']}\n{r['dispatch']}" for r in sel]
        ax.bar(names, [float(r["latency_ms"]) for r in sel])
        ax.set_ylabel("ms / forward"); ax.set_title("Inference latency (E=8, B=32, T=128)")
        plt.xticks(rotation=45, ha="right", fontsize=8); fig.tight_layout()
        fig.savefig(f"{R}/latency.png", dpi=150)
        lines.append("\n![latency](latency.png)\n")

    if os.path.exists(f"{R}/breakdown.json"):
        bd = json.load(open(f"{R}/breakdown.json"))
        lines += ["## Where MoE time goes (% of total device time in forward, E=8, B=32, T=128)\n",
                  "```", json.dumps(bd, indent=2), "```"]

    if os.path.exists(f"{R}/router_bench.csv"):
        rows = list(csv.DictReader(open(f"{R}/router_bench.csv")))
        lines += ["\n## Router optimisation (batched dispatch, seq 128)\n",
                  "| Mode | Variant | E | Batch | Dense ms | Original router ms | Fast router ms | Speed-up | Slowdown vs dense (orig → fast) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['mode']} | {r['variant']} | {r['n_experts']} | {r['batch']} | {r['dense_ms']} | "
                         f"{r['orig_ms']} | {r['fast_ms']} | {r['speedup']}x | "
                         f"{r['slowdown_vs_dense_orig']}x → {r['slowdown_vs_dense_fast']}x |")
    if os.path.exists(f"{R}/router_breakdown.json"):
        lines += ["\n## Time breakdown, original vs fast router (% of forward, E=8, B=32)\n", "```",
                  json.dumps(json.load(open(f"{R}/router_breakdown.json")), indent=2), "```"]

    open(f"{R}/summary.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
