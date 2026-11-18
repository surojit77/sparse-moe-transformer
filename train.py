"""
Train one configuration and write results/<name>.json

Examples
  python train.py --name dense                --ffn dense --d_ff 1024
  python train.py --name top1_e8              --ffn moe --top_k 1 --n_experts 8 --d_ff 1024
  python train.py --name top2_e8_matched      --ffn moe --top_k 2 --n_experts 8 --d_ff 512
"""
import argparse
import json
import math
import os
import random
import time
from contextlib import nullcontext

import numpy as np
import torch

from moe.data import amp_dtype_for, eval_batches, get_splits, random_batch
from moe.model import ModelConfig, Transformer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True)
    p.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "synthetic"])
    p.add_argument("--ffn", default="dense", choices=["dense", "moe"])
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=1)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=4)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--capacity_factor", type=float, default=1.25)
    p.add_argument("--aux_loss_coef", type=float, default=0.01)
    p.add_argument("--dispatch", default="batched", choices=["naive", "sorted", "batched"])
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out_dir", default="results")
    p.add_argument("--save_ckpt", action="store_true")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = (step - a.warmup) / max(1, a.steps - a.warmup)
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))


@torch.no_grad()
def evaluate(model, data, a, amp_ctx):
    model.eval()
    tot_loss, tot_tok = 0.0, 0
    for x, y in eval_batches(data, a.batch_size, a.seq_len, a.device):
        with amp_ctx:
            _, loss, _, _ = model(x, y)
        tot_loss += loss.item() * y.numel()
        tot_tok += y.numel()
    model.train()
    return math.exp(tot_loss / tot_tok)


def main():
    a = parse()
    set_seed(a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    splits, vocab = get_splits(a.dataset)

    cfg = ModelConfig(vocab_size=vocab, seq_len=a.seq_len, n_layer=a.n_layer, n_head=a.n_head,
                      d_model=a.d_model, d_ff=a.d_ff, dropout=a.dropout, ffn_type=a.ffn,
                      n_experts=a.n_experts, top_k=a.top_k, capacity_factor=a.capacity_factor,
                      aux_loss_coef=a.aux_loss_coef, dispatch=a.dispatch)
    model = Transformer(cfg).to(a.device)
    pc = model.param_counts()
    print(f"[{a.name}] params total={pc['total']/1e6:.2f}M active={pc['active']/1e6:.2f}M "
          f"(non-emb total={pc['total_non_emb']/1e6:.2f}M active={pc['active_non_emb']/1e6:.2f}M)")

    # Mixed precision: bf16 on Ampere+, fp16 + GradScaler on T4/V100, fp32 on CPU
    use_cuda = a.device.startswith("cuda")
    amp_dtype = amp_dtype_for(a.device)
    scaler = torch.amp.GradScaler("cuda") if amp_dtype == torch.float16 else None
    print(f"[{a.name}] device={a.device} amp={amp_dtype}")
    amp_ctx = torch.autocast("cuda", dtype=amp_dtype) if amp_dtype else nullcontext()

    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95))

    gen = torch.Generator().manual_seed(a.seed)
    history, best_val, best_state = [], float("inf"), None
    t0 = time.time(); tokens = 0
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        x, y = random_batch(splits["train"], a.batch_size, a.seq_len, a.device, gen)
        with amp_ctx:
            _, lm_loss, aux_loss, stats = model(x, y)
            loss = lm_loss + aux_loss
        opt.zero_grad(set_to_none=True)
        if scaler:
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        tokens += x.numel()

        if step % a.log_every == 0 or step == a.steps - 1:
            moe_stats = [s for s in stats if s]
            msg = f"step {step:5d} lm_loss {lm_loss.item():.3f}"
            if moe_stats:
                bal = sum(s["aux"].item() for s in moe_stats) / len(moe_stats)
                drop = sum(s["drop_frac"].item() for s in moe_stats) / len(moe_stats)
                msg += f" balance {bal:.3f} (1.0=perfect) drop {drop*100:.1f}%"
            print(msg, flush=True)

        if (step + 1) % a.eval_every == 0 or step == a.steps - 1:
            val_ppl = evaluate(model, splits["validation"], a, amp_ctx)
            history.append({"step": step + 1, "val_ppl": val_ppl, "train_loss": lm_loss.item()})
            print(f"  eval step {step+1}: val ppl {val_ppl:.2f}", flush=True)
            if val_ppl < best_val:
                best_val = val_ppl
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    if use_cuda:
        torch.cuda.synchronize()
    train_time = time.time() - t0
    model.load_state_dict(best_state)                      # early-stopping on val
    test_ppl = evaluate(model, splits["test"], a, amp_ctx)

    # Expert utilisation on validation data (per layer)
    load = []
    model.eval()
    with torch.no_grad():
        x, _ = next(eval_batches(splits["validation"], 64, a.seq_len, a.device))
        _, _, _, stats = model(x)
        for s in stats:
            if s:
                load.append([round(v, 4) for v in s["load"].tolist()])

    res = {
        "name": a.name, "args": vars(a), "config": cfg.to_dict(), "params": pc,
        "flops_per_token": model.flops_per_token(), "best_val_ppl": best_val,
        "test_ppl": test_ppl, "history": history, "train_time_s": train_time,
        "train_tokens_per_s": tokens / train_time, "expert_load": load,
    }
    path = os.path.join(a.out_dir, f"{a.name}.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=2)
    if a.save_ckpt:
        torch.save({"config": cfg.to_dict(), "state_dict": best_state},
                   os.path.join(a.out_dir, f"{a.name}.pt"))
    print(f"[{a.name}] best val ppl {best_val:.2f} | test ppl {test_ppl:.2f} | "
          f"{res['train_tokens_per_s']:.0f} tok/s -> {path}")


if __name__ == "__main__":
    main()
