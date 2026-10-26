import torch
from latent_diffusion.models import LatentDiffusion, ModelConfig
from latent_diffusion.diffusion import corrupt

def test_recurrent_transformer_has_no_gru():
    cfg = ModelConfig(d_model=32, n_heads=4, decoder_layers=1,
                      latent_basis=32, reasoning_steps=2)
    model = LatentDiffusion(cfg)
    assert not any(isinstance(m, (torch.nn.GRU, torch.nn.GRUCell))
                   for m in model.reasoner.modules())
    states = model.reasoner(torch.tensor([[1, 2], [3, 4]]))
    assert states.shape == (2, 3, 32)

def test_full_mask_probability_and_timestep():
    target = torch.tensor([[1, 2], [3, 4]])
    noisy, mask, t = corrupt(target, 10, 11, full_mask_prob=1.0)
    assert mask.all()
    assert noisy.eq(10).all()
    assert t.eq(1).all()
