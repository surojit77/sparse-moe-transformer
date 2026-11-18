"""Correctness checks.  Run:  python -m tests.test_moe"""
import torch

from moe.model import ModelConfig, MoEFFN, Transformer


def _layer(dispatch, k=2, E=4, cf=1.25):
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=32, d_ff=64, n_experts=E, top_k=k, ffn_type="moe",
                      dispatch=dispatch, dropout=0.0, capacity_factor=cf)
    return MoEFFN(cfg)


def test_dispatch_equivalence():
    """naive, sorted and batched must give identical outputs AND gradients."""
    for k in (1, 2):
        for cf in (0.5, 1.25, 4.0):                  # 0.5 forces token dropping
            x = torch.randn(4, 16, 32)
            outs, grads = [], []
            for d in ("naive", "sorted", "batched"):
                m = _layer(d, k=k, cf=cf)
                xi = x.clone().requires_grad_(True)
                y, loss, _ = m(xi)
                (y.pow(2).sum() + loss).backward()
                outs.append(y.detach()); grads.append(m.w1.grad.clone())
            for o, g in zip(outs[1:], grads[1:]):
                assert torch.allclose(outs[0], o, atol=1e-5), (k, cf)
                assert torch.allclose(grads[0], g, atol=1e-5), (k, cf)
    print("ok  dispatch equivalence (outputs + grads, with and without dropping)")


def test_capacity_drops():
    m = _layer("batched", k=1, cf=0.25)
    _, _, s = m(torch.randn(2, 32, 32))
    assert s["drop_frac"] > 0
    m = _layer("batched", k=1, cf=8.0)
    _, _, s = m(torch.randn(2, 32, 32))
    assert s["drop_frac"] == 0
    print("ok  capacity factor drops tokens only when over capacity")


def test_balance_loss_uniform():
    """With a zero router, probs are uniform -> balance loss == 1.0 exactly."""
    m = _layer("batched", k=1, E=4)
    torch.nn.init.zeros_(m.router.weight)
    _, _, s = m(torch.randn(2, 64, 32))
    assert abs(s["aux"].item() - 1.0) < 1e-5, s["aux"].item()
    print("ok  balance loss = 1.0 for uniform router")


def test_param_and_flop_accounting():
    d = Transformer(ModelConfig(vocab_size=100, ffn_type="dense", d_ff=1024))
    t1 = Transformer(ModelConfig(vocab_size=100, ffn_type="moe", top_k=1, n_experts=8, d_ff=1024))
    t2 = Transformer(ModelConfig(vocab_size=100, ffn_type="moe", top_k=2, n_experts=8, d_ff=512))
    # matched configs: same active params except tiny router
    for m in (t1, t2):
        diff = m.param_counts()["active"] - d.param_counts()["active"]
        assert diff == 4 * 256 * 8, diff                 # router weights only
    assert t1.param_counts()["total"] > d.param_counts()["total"]
    print("ok  matched-compute configs have equal active params (+router)")


def test_full_model_backward():
    m = Transformer(ModelConfig(vocab_size=100, seq_len=16, ffn_type="moe", top_k=2, n_experts=4,
                                d_model=32, n_head=2, d_ff=64))
    x = torch.randint(0, 100, (2, 16))
    _, lm, aux, _ = m(x, x)
    (lm + aux).backward()
    assert m.blocks[0].ffn.router.weight.grad.abs().sum() > 0
    print("ok  router receives gradient")


def test_fast_router_equivalence():
    """Sort-based positions must equal the one-hot/cumsum positions exactly."""
    for k in (1, 2):
        for E in (2, 8, 16):
            m = _layer("batched", k=k, E=E)
            eid = torch.randint(0, E, (512, k))
            m.cfg.fast_router = False; a = m._positions(eid)[1]
            m.cfg.fast_router = True;  b = m._positions(eid)[1]
            assert torch.equal(a, b), (k, E)
    # full layer: same outputs in training mode (losses identical too)
    x = torch.randn(4, 16, 32)
    m = _layer("batched", k=2, cf=0.5)
    m.cfg.fast_router = False; y1, l1, _ = m(x)
    m.cfg.fast_router = True;  y2, l2, _ = m(x)
    assert torch.allclose(y1, y2) and torch.allclose(l1, l2)
    print("ok  fast router == original router (positions, outputs, losses)")


if __name__ == "__main__":
    test_fast_router_equivalence()
    test_dispatch_equivalence()
    test_capacity_drops()
    test_balance_loss_uniform()
    test_param_and_flop_accounting()
    test_full_model_backward()
