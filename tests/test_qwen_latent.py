"""Cache / mask correctness for QwenLatentDiffusion. Needs the Qwen3-0.6B weights, so opt in:
    LDLAB_QWEN_TESTS=1 pytest tests/test_qwen_latent.py
Runs in fp32 so that cached vs uncached differences are numerical noise, not bf16 rounding."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("LDLAB_QWEN_TESTS"), reason="set LDLAB_QWEN_TESTS=1 (downloads Qwen3-0.6B)")
TOL = 2e-2  # absolute, on final-norm hidden states


@pytest.fixture(scope="module")
def setup():
    from latent_diffusion.qwen_latent import QwenLatentDiffusion
    torch.manual_seed(0)
    m = QwenLatentDiffusion(steps=3, lora_r=0, dtype=torch.float32).eval()
    batch = m.tok(["What is 2+3?", "A much longer prompt: what is twelve times eleven, minus seven?"],
                  return_tensors="pt", padding=True)  # left padded, rows have different lengths
    return m, batch.input_ids, batch.attention_mask


def _full_forward(m, embeds, mask4d, pos):
    return m._forward(embeds, pos, mask4d, None, 1)


def test_cached_latent_steps_match_uncached(setup):
    m, ids, am = setup
    with torch.no_grad():
        _, _, _, latents, h_cached = m.prefix(ids, am)
        r = len(latents)
        embeds = torch.cat([m._embed(ids)] + latents, dim=1)
        mask = torch.cat([am, am.new_ones(len(am), r)], dim=1)
        pos = torch.cat([(am.cumsum(-1) - 1).clamp(min=0), am.sum(-1)[:, None] + torch.arange(r)], dim=1)
        h_ref = m._forward(embeds, pos, mask, None, 1)[:, -1]
    assert (h_cached - h_ref).abs().max() < TOL


def _reference_canvas(m, ids, am, latents, tokens):
    """Uncached full-sequence forward with an explicit 4D mask: causal prefix, canvas sees all non-pad prefix + canvas."""
    b, a, r = len(ids), tokens.shape[1], len(latents)
    p = ids.shape[1] + r
    emb = torch.cat([m._embed(ids)] + latents, dim=1)
    cemb = torch.where(tokens.eq(m.mask_id)[..., None], m.mask_vec, m._embed(tokens))
    pm = torch.cat([am, am.new_ones(b, r)], dim=1).bool()
    allowed = torch.zeros(b, p + a, p + a, dtype=torch.bool)
    allowed[:, :p, :p] = torch.tril(torch.ones(p, p, dtype=torch.bool))[None] & pm[:, None, :]
    allowed[:, p:, :p] = pm[:, None, :]
    allowed[:, p:, p:] = True
    bias = torch.zeros(b, 1, p + a, p + a).masked_fill(~allowed[:, None], torch.finfo(torch.float32).min)
    n = am.sum(-1)
    pos = torch.cat([(am.cumsum(-1) - 1).clamp(min=0), n[:, None] + torch.arange(r),
                     (n + r)[:, None] + torch.arange(a)], dim=1)
    return m.lm(inputs_embeds=torch.cat([emb, cemb], 1), position_ids=pos, attention_mask=bias,
                use_cache=False, output_hidden_states=True, logits_to_keep=1).hidden_states[-1][:, -a:]


def test_canvas_matches_uncached_reference_and_cache_is_reusable(setup):
    m, ids, am = setup
    tokens = torch.tensor([[m.mask_id, 5, m.mask_id, 7], [m.mask_id] * 4])
    with torch.no_grad():
        cache, pmask, start, latents, _ = m.prefix(ids, am)
        plen = pmask.shape[1]
        first = m.canvas_hidden(cache, pmask, start, tokens)
        cache.crop(plen)
        second = m.canvas_hidden(cache, pmask, start, tokens)  # same cache after crop
        ref = _reference_canvas(m, ids, am, latents, tokens)
    assert (first - ref).abs().max() < TOL
    assert (first - second).abs().max() < 1e-4


def test_canvas_is_bidirectional_but_prefix_never_sees_it(setup):
    m, ids, am = setup
    a = torch.tensor([[m.mask_id, m.mask_id, m.mask_id, 7]] * 2)
    b = a.clone(); b[:, 3] = 9  # change only the LAST canvas token
    with torch.no_grad():
        cache, pmask, start, *_ = m.prefix(ids, am)
        plen = pmask.shape[1]
        key_before = cache.layers[0].keys.clone() if hasattr(cache, "layers") else cache.key_cache[0].clone()
        ha = m.canvas_hidden(cache, pmask, start, a); cache.crop(plen)
        hb = m.canvas_hidden(cache, pmask, start, b); cache.crop(plen)
        key_after = cache.layers[0].keys if hasattr(cache, "layers") else cache.key_cache[0]
    assert (ha[:, 0] - hb[:, 0]).abs().max() > 1e-3  # first canvas position reacts to a later token
    assert torch.equal(key_before, key_after)         # prefix KV untouched by canvas passes
