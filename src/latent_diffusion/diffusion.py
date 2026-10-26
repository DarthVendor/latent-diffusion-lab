"""Discrete mask corruption and confidence-based iterative remasking."""
import math
import torch
from torch.nn import functional as F

def corrupt(target, mask_id, pad_id):
    b, _ = target.shape
    valid = target.ne(pad_id)
    if not valid.any(dim=-1).all():
        raise ValueError('every target needs at least one non-padding token')
    t = torch.rand(b, device=target.device).clamp(min=1e-3)
    masks = (torch.rand(target.shape, device=target.device) < t[:, None]) & valid
    force_full = torch.rand(b, device=target.device) < full_mask_prob
    masks[force_full] = valid[force_full]
    for i in range(b):
        if not masks[i].any():
            available = valid[i].nonzero(as_tuple=True)[0]
            j = available[torch.randint(len(available), (1,), device=target.device)]
            masks[i, j] = True
    t = masks.float().sum(-1) / valid.float().sum(-1)
    return target.masked_fill(masks, mask_id), masks, t

def masked_loss(logits, target, masks, t=None, weighted=False):
    raw = F.cross_entropy(logits.transpose(1, 2), target, reduction='none')
    if not weighted:
        return raw[masks].mean()
    weights = masks.float() / t[:, None].clamp(min=1e-3)
    return (raw * weights).sum() / weights.sum().clamp_min(1)

@torch.no_grad()
def sample(model, prompt, output_length, steps=8, temperature=1.0, stochastic_reasoning=False):
    if steps < 1 or output_length < 1:
        raise ValueError('steps and output_length must be >=1')
    memory = model.encode(prompt, stochastic=stochastic_reasoning)
    y = torch.full((len(prompt), output_length), model.cfg.mask_id,
                   dtype=torch.long, device=prompt.device)
    for step in range(steps):
        t = torch.full((len(prompt),), (steps - step) / steps, device=prompt.device)
        logits = model.decoder(y, t, memory)
        logits[..., 10:] = -1e9
        probs = F.softmax(logits / max(temperature, 1e-6), dim=-1)
        conf, predicted = probs.max(dim=-1)
        candidate = torch.where(y.eq(model.cfg.mask_id), predicted, y)
        remaining = int(math.floor(output_length * (1 - (step + 1) / steps)))
        if step == steps - 1 or remaining == 0:
            y = candidate
        else:
            confidence = torch.where(y.eq(model.cfg.mask_id), conf, torch.ones_like(conf))
            indices = confidence.argsort(dim=-1)[:, :remaining]
            y = candidate.scatter(1, indices, model.cfg.mask_id)
    return y
