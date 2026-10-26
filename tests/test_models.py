import torch
from latent_diffusion.models import LatentDiffusion, ModelConfig
from latent_diffusion.diffusion import corrupt, masked_loss, sample
from latent_diffusion.data import addition_data

def test_data_split():
    tr, va = addition_data()
    assert len(tr[0]) == 80 and len(va[0]) == 20
    assert torch.equal(tr[0].sum(-1), tr[1][:, 0] * 10 + tr[1][:, 1])

def test_forward_and_backward():
    cfg = ModelConfig(d_model=32, n_heads=4, latent_basis=32, decoder_layers=1,
                      reasoning_steps=2, memory_slots=2)
    model = LatentDiffusion(cfg)
    tr, _ = addition_data()
    x, y = tr[0][:4], tr[1][:4]
    noisy, mask, t = corrupt(y, cfg.mask_id, cfg.pad_id)
    logits = model(x, noisy, t, stochastic=True)
    assert logits.shape == (4, 2, cfg.vocab_size)
    loss = masked_loss(logits, y, mask)
    loss.backward()
    assert model.reasoner.project.weight.grad is not None
    assert torch.isfinite(loss)

def test_sampler_reveals_all_tokens():
    cfg = ModelConfig(d_model=32, n_heads=4, latent_basis=32,
                      decoder_layers=1, reasoning_steps=2, memory_slots=2)
    model = LatentDiffusion(cfg).eval()
    output = sample(model, torch.tensor([[1, 2], [3, 4]]), 2, steps=3)
    assert output.shape == (2, 2)
    assert ((output >= 0) & (output <= 9)).all()

def test_no_latent_ablation():
    cfg = ModelConfig(d_model=32, n_heads=4, decoder_layers=1, use_latent=False)
    model = LatentDiffusion(cfg)
    assert model.reasoner(torch.tensor([[1, 2]])).shape[1] == 1
