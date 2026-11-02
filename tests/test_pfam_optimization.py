"""Tests for optimized Pfam-set training configuration."""
import math

import pytest
import torch
from torch import nn

from newkcat.pfam_fusion.cli import build_optimizer, metrics, scheduled_lr
from newkcat.pfam_fusion.model import PfamCataPro, PfamEncoder


def test_scheduler_warmup_and_cosine_end():
    total = 100
    assert scheduled_lr(1, total, 20, 5e-5, 0.1) == pytest.approx(2.5e-6)
    assert scheduled_lr(20, total, 20, 5e-5, 0.1) == pytest.approx(5e-5)
    assert scheduled_lr(100, total, 20, 5e-5, 0.1) == pytest.approx(5e-6)
    assert scheduled_lr(60, total, 20, 5e-5, 0.1) > 5e-6
    assert scheduled_lr(60, total, 20, 5e-5, 0.1) < 5e-5


def test_full_dataset_metrics():
    y = [1., 2., 3., 4.]
    p = [1., 3., 2., 4.]
    m = metrics(y, p)
    assert m["loss"] == pytest.approx(.5)
    assert m["rmse"] == pytest.approx(math.sqrt(.5))
    assert m["mae"] == pytest.approx(.5)
    assert m["r2"] == pytest.approx(.6)
    assert m["pcc"] == pytest.approx(.8)
    assert m["loss"] == pytest.approx(m["rmse"] ** 2)


def test_xavier_pad_zero_and_fc_dropout():
    torch.manual_seed(7)
    m = PfamEncoder(
        vocab_size=8, nmax=4, ordered=False,
        dropout=0.1, fc_dropout=0.1, init_scheme="xavier",
    )
    assert isinstance(m.out[-1], nn.Dropout)
    assert m.out[-1].p == pytest.approx(.1)
    assert torch.count_nonzero(m.embedding.weight[0]) == 0
    x = torch.tensor([[2, 3, 0, 0], [0, 0, 0, 0]])
    m.eval()
    p = m(x)
    assert p.shape == (2, 256)
    assert torch.count_nonzero(p[1]) == 0
    assert torch.allclose(p, m(x))
    m.train()
    loss = m(x).square().sum()
    loss.backward()
    assert torch.isfinite(m.embedding.weight.grad).all()


def test_optimizer_excludes_norm_and_bias_decay():
    model = PfamCataPro(
        vocab_size=8, nmax=4, ordered=False,
        dropout=.1, fc_dropout=.1, init_scheme="xavier",
    )
    opt = build_optimizer(model, 5e-5, 1e-3)
    assert [g["weight_decay"] for g in opt.param_groups] == [1e-3, 0.0]
    assert opt.defaults["lr"] == pytest.approx(5e-5)
    decayed = {id(p) for p in opt.param_groups[0]["params"]}
    assert id(model.pfam.out[0].weight) in decayed
    assert id(model.pfam.out[1].weight) not in decayed
    assert id(model.catapro.prot_norm.weight) not in decayed
