"""Cache / mask correctness for QwenLatentDiffusion. Needs the Qwen3-0.6B weights, so opt in:
    LDLAB_QWEN_TESTS=1 pytest tests/test_qwen_latent.py
Runs in fp32 so that cached vs uncached differences are numerical noise, not bf16 rounding."""
import os

import pytest
import torch

pytestmark = pytest.mark.skipif(not os.environ.get("LDLAB_QWEN_TESTS"), reason="set LDLAB_QWEN_TESTS=1 (downloads Qwen3-0.6B)")
TOL = 2e-3  # absolute, on final-norm hidden states (scale ~50; observed diffs ~2e-4)


@pytest.fixture(scope="module", params=["mixture", "hidden"])
def setup(request):
    from latent_diffusion.qwen_latent import QwenLatentDiffusion
    torch.manual_seed(0)
    m = QwenLatentDiffusion(steps=3, lora_r=0, dtype=torch.float32, latent_mode=request.param).eval()
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


def _reference_canvas(m, ids, am, latents, tokens, latents_only=False):
    """Uncached full-sequence forward with an explicit 4D mask: causal prefix, canvas sees all non-pad prefix + canvas."""
    b, a, r = len(ids), tokens.shape[1], len(latents)
    p = ids.shape[1] + r
    emb = torch.cat([m._embed(ids)] + latents, dim=1)
    cemb = torch.where(tokens.eq(m.mask_id)[..., None], m.mask_vec, m._embed(tokens))
    pm = torch.cat([am, am.new_ones(b, r)], dim=1).bool()
    allowed = torch.zeros(b, p + a, p + a, dtype=torch.bool)
    allowed[:, :p, :p] = torch.tril(torch.ones(p, p, dtype=torch.bool))[None] & pm[:, None, :]
    allowed[:, p:, :p] = pm[:, None, :]
    if latents_only:
        allowed[:, p:, :p - r] = False  # canvas rows cannot see the prompt tokens
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


def test_aux_solution_loss_matches_uncached_reference_and_hides_prompt(setup):
    """Solution tokens see only the latent positions (never the prompt) + earlier solution tokens."""
    import torch.nn.functional as F
    m, ids, am = setup
    sol = torch.tensor([[11, 12, 13, 14, 15], [21, 22, 23, 0, 0]])
    sm = torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]])
    with torch.no_grad():
        cache, pmask, start, latents, h_last = m.prefix(ids, am)
        got = m.aux_solution_loss(cache, pmask, start, h_last, sol, sm)
        # reference: one uncached forward over [prompt | latents | sol[:, :-1]] with an explicit 4D mask
        b, r, s = len(ids), len(latents), sol.shape[1]
        p = ids.shape[1]
        emb = torch.cat([m._embed(ids)] + latents + [m._embed(sol[:, :-1])], dim=1)
        n = p + r + s - 1
        pm = torch.cat([am, am.new_ones(b, r)], dim=1).bool()
        allowed = torch.zeros(b, n, n, dtype=torch.bool)
        allowed[:, :p + r, :p + r] = torch.tril(torch.ones(p + r, p + r, dtype=torch.bool))[None] & pm[:, None, :]
        allowed[:, p + r:, p:p + r] = True
        allowed[:, p + r:, p + r:] = torch.tril(torch.ones(s - 1, s - 1, dtype=torch.bool))
        bias = torch.zeros(b, 1, n, n).masked_fill(~allowed[:, None], torch.finfo(torch.float32).min)
        pos = torch.cat([(am.cumsum(-1) - 1).clamp(min=0), am.sum(-1)[:, None] + torch.arange(r + s - 1)], dim=1)
        hid = m.lm(inputs_embeds=emb, position_ids=pos, attention_mask=bias, use_cache=False,
                   output_hidden_states=True, logits_to_keep=1).hidden_states[-1]
        head = m.lm.get_output_embeddings()
        first = F.cross_entropy(head(hid[:, p + r - 1])[:, :m.vocab], sol[:, 0], reduction="none")
        rest = F.cross_entropy(head(hid[:, p + r:])[..., :m.vocab].transpose(1, 2), sol[:, 1:], reduction="none")
        ce = torch.cat([first[:, None], rest], dim=1)
        want = (ce * sm).sum() / sm.sum()
    assert abs(got.item() - want.item()) < 1e-2


def test_latents_only_canvas_cannot_see_the_prompt(setup):
    m, ids, am = setup
    tokens = torch.tensor([[m.mask_id, 5, m.mask_id, 7], [m.mask_id] * 4])
    with torch.no_grad():
        cache, pmask, start, latents, _ = m.prefix(ids, am)
        m.canvas_latents_only = True
        try:
            got = m.canvas_hidden(cache, pmask, start, tokens)
            ref = _reference_canvas(m, ids, am, latents, tokens, latents_only=True)
            cache.crop(pmask.shape[1])
            m.canvas_latents_only = False
            full = m.canvas_hidden(cache, pmask, start, tokens)
        finally:
            m.canvas_latents_only = False
    assert (got - ref).abs().max() < TOL      # matches an uncached run where canvas rows skip the prompt
    assert (got - full).abs().max() > 1e-2    # and really differs from seeing the prompt


def test_adapter_off_prefix_cache_equals_base_model_cache():
    """With prefix_adapter_off the drafter's prefix KV must be exactly the un-adapted model's, even with a non-zero LoRA."""
    from latent_diffusion.qwen_latent import QwenLatentDiffusion
    torch.manual_seed(0)
    base = QwenLatentDiffusion(steps=0, lora_r=0, dtype=torch.float32).eval()
    drafter = QwenLatentDiffusion(steps=0, lora_r=4, dtype=torch.float32, prefix_adapter_off=True).eval()
    for n, p in drafter.named_parameters():
        if "lora_B" in n:
            p.data.normal_(0, 0.5)  # make the adapter clearly non-trivial
    batch = base.tok(["What is 2+3?", "A longer prompt about twelve times eleven"], return_tensors="pt", padding=True)
    with torch.no_grad():
        c0, pm, st, *_ = base.prefix(batch.input_ids, batch.attention_mask)
        c1, pm1, st1, *_ = drafter.prefix(batch.input_ids, batch.attention_mask)
        key = lambda c: c.layers[-1].keys if hasattr(c, "layers") else c.key_cache[-1]
        assert (key(c0) - key(c1)).abs().max() < 1e-4
        tokens = torch.full((2, 4), base.mask_id)
        h_base = base.canvas_hidden(c0, pm, st, tokens)
        h_draft = drafter.canvas_hidden(c1, pm1, st1, tokens)
    assert (h_base - h_draft).abs().max() > 1e-2  # the adapter really acts on the canvas tokens
