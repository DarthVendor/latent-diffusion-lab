"""Static-cache, torch.compile(reduce-overhead) inference path for QwenLatentDiffusion: the same trick as the compiled AR
baseline. Prefill, one latent step and one canvas pass are three fixed-shape functions, so every (latent steps, denoise
steps) config reuses the same three graphs. Slots are fixed: [padded prompt P | latents | canvas A]; the canvas KV is simply
overwritten each pass, so there is no cache crop. Inference only: it merges the LoRA into the weights (mutates `m`).
"""
import torch
import torch.nn.functional as F
from transformers import StaticCache


class CompiledLatentDiffusion:
    def __init__(self, m, batch, plen, canvas=12, max_latent=4, compile=True):
        if hasattr(m.lm, "merge_and_unload"):
            m.lm = m.lm.merge_and_unload()
        self.m, self.B, self.P, self.A = m, batch, plen, canvas
        self.L = plen + max_latent + canvas
        dev, self.dt = m.device, m.lm.dtype
        self.cache = StaticCache(config=m.lm.config, max_batch_size=batch, max_cache_len=self.L, device=dev, dtype=self.dt)
        self.ar = torch.arange(self.L, device=dev)
        self.slot = [torch.tensor([i], device=dev) for i in range(self.L)]
        wrap = (lambda f: torch.compile(f, mode="reduce-overhead", fullgraph=True)) if compile else (lambda f: f)
        self._prefill, self._latent, self._canvas = wrap(self._prefill_impl), wrap(self._latent_impl), wrap(self._canvas_impl)

    def _run(self, emb, pos, vis, slots):
        bias = torch.zeros(vis.shape[0], 1, vis.shape[1], self.L, dtype=self.dt, device=emb.device)
        bias = bias.masked_fill(~vis[:, None], torch.finfo(self.dt).min)
        return self.m.lm.model(inputs_embeds=emb, position_ids=pos, attention_mask=bias, past_key_values=self.cache,
                               cache_position=slots, use_cache=True).last_hidden_state

    def _prefill_impl(self, ids, pm):
        slots = self.ar[:self.P]
        vis = (slots[:, None] >= self.ar[None, :])[None] & F.pad(pm.bool(), (0, self.L - self.P))[:, None, :]
        return self._run(self.m._embed(ids), (pm.cumsum(-1) - 1).clamp(min=0), vis, slots)[:, -1]

    def _latent_impl(self, h, pos, pm_full, slot):
        m, hf = self.m, h.float()
        if m.latent_mode == "hidden":
            e = m.thought(hf / hf.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()) * m.emb_rms
        else:
            e = F.softmax(m.project(m.norm(hf)) / m.tau, dim=-1) @ m.basis
        vis = pm_full | ((self.ar >= self.P) & (self.ar <= slot))[None]
        return self._run(e.to(h.dtype)[:, None], pos, vis[:, None, :], slot)[:, -1]

    def _canvas_impl(self, y, vis_prefix, start, slot0, remaining):
        m, a = self.m, self.A
        emb = m._embed(y)
        emb = torch.where(y.eq(m.mask_id)[..., None], m.mask_vec.to(emb.dtype), emb)
        vis = (vis_prefix | ((self.ar >= slot0) & (self.ar < slot0 + a))[None])[:, None, :].expand(-1, a, -1)
        h = self._run(emb, start[:, None] + self.ar[:a], vis, slot0 + self.ar[:a])
        conf, pred = F.softmax(m.lm.get_output_embeddings()(h)[..., :m.vocab].float(), dim=-1).max(-1)
        is_mask = y.eq(m.mask_id)
        cand = torch.where(is_mask, pred, y)
        idx = torch.where(is_mask, conf, torch.ones_like(conf)).argsort(-1)
        remask = self.ar[:a][None] < remaining      # the `remaining` least-confident positions go back to mask
        return cand.scatter(1, idx, torch.where(remask, m.mask_id, cand.gather(1, idx)))

    @torch.no_grad()
    def generate(self, ids, mask, latent_steps, denoise_steps):
        m, P, A, R, D = self.m, self.P, self.A, latent_steps, denoise_steps
        assert ids.shape == (self.B, P) and R <= self.L - P - A
        assert not (m.canvas_latents_only and R == 0)
        step_begin = torch.compiler.cudagraph_mark_step_begin
        n, pm_full = mask.sum(-1), F.pad(mask.bool(), (0, self.L - P))
        step_begin(); h = self._prefill(ids, mask).clone()
        for r in range(R):
            step_begin(); h = self._latent(h, (n + r)[:, None], pm_full, self.slot[P + r]).clone()
        lat = ((self.ar >= P) & (self.ar < P + R))[None]
        vis = lat if m.canvas_latents_only else lat | pm_full
        y = torch.full((self.B, A), m.mask_id, dtype=torch.long, device=ids.device)
        for s in range(D):
            rem = 0 if s == D - 1 else int(A * (1 - (s + 1) / D))
            step_begin(); y = self._canvas(y, vis, n + R, self.slot[P + R], torch.tensor([rem], device=ids.device)).clone()
        return y
