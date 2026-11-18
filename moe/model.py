"""
Decoder-only Transformer with Dense / Top-1 / Top-2 Sparse Mixture-of-Experts FFNs.

Key pieces (all things interviewers ask about):
  * Router in fp32, softmax over experts, top-k selection
  * Switch/GShard-style load-balancing auxiliary loss
  * Router z-loss (keeps router logits small -> stable training)
  * Expert capacity + token dropping (capacity_factor)
  * Three dispatch implementations with IDENTICAL math, different speed:
        naive   : Python loop over experts, boolean mask per expert (the slow baseline)
        sorted  : sort tokens by expert once, loop over contiguous slices
        batched : scatter tokens into an [E, C, d] buffer and run ONE batched GEMM
                  (bmm) for all experts -> no Python loop, no host<->device sync
"""
import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import record_function


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    seq_len: int = 128
    n_layer: int = 4
    n_head: int = 4
    d_model: int = 256
    d_ff: int = 1024            # hidden size of the dense FFN, or of EACH expert
    dropout: float = 0.1
    ffn_type: str = "dense"     # "dense" | "moe"
    n_experts: int = 8
    top_k: int = 1              # 1 = Switch-style, 2 = GShard-style
    capacity_factor: float = 1.25
    eval_capacity_factor: float = 2.0
    aux_loss_coef: float = 0.01
    z_loss_coef: float = 1e-3
    dispatch: str = "batched"   # "naive" | "sorted" | "batched"
    fast_router: bool = True    # sort-based slot positions + skip router losses at inference
    moe_every: int = 1          # put an MoE FFN in every n-th layer (1 = all layers)

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
#  Attention / dense FFN
# --------------------------------------------------------------------------- #
class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.dropout = cfg.dropout
        self.resid_drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class DenseFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.d_model, cfg.d_ff, bias=False)
        self.fc2 = nn.Linear(cfg.d_ff, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        B, T, C = x.shape
        y = self.drop(self.fc2(F.gelu(self.fc1(x))))
        return y, x.new_zeros(()), {}


# --------------------------------------------------------------------------- #
#  Sparse MoE FFN
# --------------------------------------------------------------------------- #
class MoEFFN(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        E, d, h = cfg.n_experts, cfg.d_model, cfg.d_ff
        assert 1 <= cfg.top_k <= E
        self.router = nn.Linear(d, E, bias=False)
        # Expert weights stored stacked so every dispatch mode uses the same params.
        self.w1 = nn.Parameter(torch.empty(E, d, h))
        self.w2 = nn.Parameter(torch.empty(E, h, d))
        nn.init.normal_(self.w1, std=0.02)
        nn.init.normal_(self.w2, std=0.02 / math.sqrt(2 * cfg.n_layer))
        nn.init.normal_(self.router.weight, std=0.02)
        self.drop = nn.Dropout(cfg.dropout)

    # ---- routing -------------------------------------------------------- #
    def _route(self, x):
        """x: [N, d] -> expert ids [N,k], gate weights [N,k], losses, stats"""
        cfg = self.cfg
        E, k = cfg.n_experts, cfg.top_k
        logits = self.router(x).float()                    # router in fp32
        probs = logits.softmax(dim=-1)                      # [N, E]
        gate, eid = probs.topk(k, dim=-1)                   # [N, k]
        if k > 1:                                           # GShard: renormalise top-k
            gate = gate / gate.sum(dim=-1, keepdim=True)

        N = x.shape[0]
        counts = torch.bincount(eid.reshape(-1), minlength=E).float()
        if cfg.fast_router and not torch.is_grad_enabled():
            # Inference: the aux / z losses are never used, so don't compute them.
            zero = logits.new_zeros(())
            return eid, gate, zero, zero, counts
        # Load-balancing loss (Switch Transformer eq. 4):  E * sum_i f_i * P_i
        # f_i = fraction of routed slots sent to expert i, P_i = mean router prob.
        f = counts / (N * k)
        P = probs.mean(dim=0)
        aux = E * torch.sum(f * P)
        # Router z-loss (ST-MoE): penalise large logits
        z = torch.logsumexp(logits, dim=-1).pow(2).mean()
        loss = cfg.aux_loss_coef * aux + cfg.z_loss_coef * z
        return eid, gate, loss, aux.detach(), counts.detach()

    def _capacity(self, N):
        cf = self.cfg.capacity_factor if self.training else self.cfg.eval_capacity_factor
        return max(1, math.ceil(cf * N * self.cfg.top_k / self.cfg.n_experts))

    def _positions(self, eid):
        """Position of each (token, choice) slot inside its expert's buffer.
        Top-1 choices of all tokens get priority over top-2 choices (GShard)."""
        N, k = eid.shape
        E = self.cfg.n_experts
        flat = eid.t().reshape(-1)                          # k-major order: [k*N]
        if self.cfg.fast_router:
            # Sort-based: O(kN log kN) memory O(kN). Stable sort keeps the k-major
            # priority order inside each expert, so results equal the one-hot version.
            order = torch.argsort(flat, stable=True)
            counts = torch.bincount(flat, minlength=E)
            starts = torch.cumsum(counts, 0) - counts       # first slot of each expert
            rank = torch.arange(flat.numel(), device=flat.device)
            pos = torch.empty_like(flat)
            pos[order] = rank - starts[flat[order]]
        else:
            # Original: builds a [kN, E] one-hot matrix and a cumsum over it.
            onehot = F.one_hot(flat, E)                     # [kN, E]
            pos = (onehot.cumsum(0) * onehot).sum(-1) - 1   # [kN]
        token = torch.arange(N, device=eid.device).repeat(k)
        return flat, pos, token

    # ---- expert MLP on a group of tokens -------------------------------- #
    def _expert(self, xs, e):
        return F.gelu(xs @ self.w1[e]) @ self.w2[e]

    # ---- three dispatch strategies -------------------------------------- #
    def _dispatch_naive(self, x, flat, gate_flat, token, keep):
        y = torch.zeros_like(x)
        for e in range(self.cfg.n_experts):
            sel = keep & (flat == e)                        # boolean mask per expert
            idx = sel.nonzero(as_tuple=True)[0]             # host sync every expert
            if idx.numel() == 0:
                continue
            tok = token[idx]
            out = self._expert(x[tok], e) * gate_flat[idx, None].to(x.dtype)
            y.index_add_(0, tok, out)
        return y

    def _dispatch_sorted(self, x, flat, gate_flat, token, keep):
        E = self.cfg.n_experts
        kept = keep.nonzero(as_tuple=True)[0]
        e_k = flat[kept]
        order = torch.argsort(e_k, stable=True)
        kept = kept[order]
        tok = token[kept]
        counts = torch.bincount(e_k, minlength=E).tolist()  # ONE host sync
        xs = x[tok]                                         # one gather
        outs, start = [], 0
        for e, c in enumerate(counts):
            if c:
                outs.append(self._expert(xs[start:start + c], e))
            start += c
        out = torch.cat(outs, 0) * gate_flat[kept, None].to(x.dtype)
        y = torch.zeros_like(x)
        y.index_add_(0, tok, out)                           # one scatter
        return y

    def _dispatch_batched(self, x, flat, gate_flat, token, pos, keep, C):
        E, d = self.cfg.n_experts, x.shape[1]
        slot = flat * C + pos.clamp(max=C - 1)              # [kN] slot in [E*C]
        with record_function("moe/dispatch"):
            buf = x.new_zeros(E * C, d)
            buf.index_copy_(0, slot[keep], x[token[keep]])
            buf = buf.view(E, C, d)
        with record_function("moe/expert_compute"):
            h = F.gelu(torch.bmm(buf, self.w1))             # [E, C, h]  one batched GEMM
            out = torch.bmm(h, self.w2).view(E * C, d)      # [E*C, d]
        with record_function("moe/combine"):
            w = (gate_flat * keep).to(x.dtype)              # dropped slots contribute 0
            y = torch.zeros_like(x)
            y.index_add_(0, token, out[slot] * w[:, None])
        return y

    def forward(self, x):
        B, T, d = x.shape
        x2 = x.reshape(-1, d)
        N = x2.shape[0]
        with record_function("moe/router"):
            eid, gate, loss, aux, counts = self._route(x2)
            C = self._capacity(N)
            flat, pos, token = self._positions(eid)
            gate_flat = gate.t().reshape(-1)
            keep = pos < C
        mode = self.cfg.dispatch
        if mode == "batched":
            y = self._dispatch_batched(x2, flat, gate_flat, token, pos, keep, C)
        else:
            with record_function(f"moe/{mode}_dispatch_and_compute"):
                if mode == "naive":
                    y = self._dispatch_naive(x2, flat, gate_flat, token, keep)
                elif mode == "sorted":
                    y = self._dispatch_sorted(x2, flat, gate_flat, token, keep)
                else:
                    raise ValueError(mode)
        stats = {
            "aux": aux,
            "load": counts / counts.sum(),                  # fraction routed to each expert
            "drop_frac": 1.0 - keep.float().mean().detach(),
        }
        return self.drop(y.view(B, T, d)), loss, stats


# --------------------------------------------------------------------------- #
#  Model
# --------------------------------------------------------------------------- #
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, use_moe: bool):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = MoEFFN(cfg) if use_moe else DenseFFN(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        y, loss, stats = self.ffn(self.ln2(x))
        return x + y, loss, stats


class Transformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg, use_moe=(cfg.ffn_type == "moe" and (i + 1) % cfg.moe_every == 0))
            for i in range(cfg.n_layer)
        )
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight             # weight tying
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        aux_total = x.new_zeros((), dtype=torch.float32)
        layer_stats = []
        for blk in self.blocks:
            x, l, s = blk(x)
            aux_total = aux_total + l
            layer_stats.append(s)
        logits = self.head(self.ln_f(x))
        lm_loss = None
        if targets is not None:
            lm_loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, lm_loss, aux_total, layer_stats

    # ---- parameter / FLOP accounting ------------------------------------ #
    def param_counts(self):
        """total params, and params actually used per token (active)."""
        cfg = self.cfg
        total = sum(p.numel() for p in self.parameters())
        expert_total, expert_active = 0, 0
        for m in self.modules():
            if isinstance(m, MoEFFN):
                per_expert = m.w1[0].numel() + m.w2[0].numel()
                expert_total += per_expert * cfg.n_experts
                expert_active += per_expert * cfg.top_k
        active = total - expert_total + expert_active
        emb = self.tok_emb.weight.numel() + self.pos_emb.weight.numel()
        return {"total": total, "active": active,
                "total_non_emb": total - emb, "active_non_emb": active - emb}

    def flops_per_token(self, T=None):
        """Forward FLOPs per token (2 * multiply-adds), analytic."""
        cfg = self.cfg
        T = T or cfg.seq_len
        d, L = cfg.d_model, cfg.n_layer
        f = 0
        for blk in self.blocks:
            f += 2 * (3 * d * d + d * d)                   # qkv + out proj
            f += 2 * 2 * T * d / 2                         # QK^T and AV, causal ~ half
            if isinstance(blk.ffn, MoEFFN):
                f += 2 * d * cfg.n_experts                  # router
                f += cfg.top_k * 2 * (2 * d * cfg.d_ff)     # k experts
            else:
                f += 2 * (2 * d * cfg.d_ff)
        f += 2 * d * cfg.vocab_size                         # LM head
        return f
