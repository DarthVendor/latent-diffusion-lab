"""Qwen3-0.6B + SLT-style continuous latent steps + bidirectional masked-diffusion answer canvas.

Layout of one sequence:  [left-padded prompt | R latent soft tokens | answer canvas of A tokens]
* prompt and latents run causally through the pretrained Qwen layers and are cached once (the prefix),
  so they can never attend to the noisy canvas;
* the canvas attends to the whole prefix and bidirectionally to itself (explicit 4D mask), reusing the
  prefix KV cache. The cache is cropped back after each denoising pass.
* LoRA on the Qwen layers, plus a trainable latent projector, basis, and mask embedding.
Latent modes: "mixture" = softmax mixture over a learned basis (collapsed to a constant in our diagnostics);
"hidden" = continuous thought: the previous step's final hidden state is rescaled to embedding scale, passed
through a learned linear map (init identity) and fed back as the next input embedding.
Aux loss: the written solution must be reconstructable from the latent positions alone (solution tokens
cannot attend to the prompt), which forces per-problem information into the latents.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from .diffusion import corrupt


class QwenLatentDiffusion(nn.Module):
    def __init__(self, name="Qwen/Qwen3-0.6B", steps=4, basis=512, tau=0.7, lora_r=16, dtype=torch.bfloat16,
                 latent_mode="mixture", canvas_latents_only=False, prefix_adapter_off=False):
        super().__init__()
        # True: the prefix (prompt + latents) runs with the LoRA adapter disabled and no grad, so its KV cache is
        # exactly the base model's. A speculative drafter can then read the TARGET's cache with no cache of its own.
        self.prefix_adapter_off = prefix_adapter_off
        assert latent_mode in ("mixture", "hidden")
        assert not canvas_latents_only or steps > 0, "a latents-only canvas needs latent steps"
        self.latent_mode = latent_mode
        # True: the answer canvas attends to the latent positions + itself, never the prompt tokens, so the
        # prompt KV can be freed once the latent steps are done (the latents must then carry the problem).
        self.canvas_latents_only = canvas_latents_only
        self.tok = AutoTokenizer.from_pretrained(name, padding_side="left")
        lm = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype, attn_implementation="sdpa")
        rows, hidden = lm.get_input_embeddings().weight.shape
        # Qwen3's embedding has spare rows past the tokenizer vocab; use the first as the mask id (no resize).
        self.vocab, self.mask_id, self.pad_id = len(self.tok), len(self.tok), self.tok.pad_token_id
        assert rows > self.mask_id, "no spare embedding row for the mask token"
        if lora_r:
            from peft import LoraConfig, get_peft_model
            lm = get_peft_model(lm, LoraConfig(r=lora_r, lora_alpha=2 * lora_r, task_type="CAUSAL_LM",
                                               target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                                               "gate_proj", "up_proj", "down_proj"]))
        else:
            lm.requires_grad_(False)
        self.lm, self.steps, self.tau = lm, steps, tau
        self._latent_hook, self._trace = None, None  # diagnostics only: edit / record latents (scripts/diagnose.py)
        emb = lm.get_input_embeddings().weight.detach()
        # latent basis starts as real token embeddings so soft tokens begin in-distribution
        self.basis = nn.Parameter(emb[torch.randperm(self.vocab)[:basis]].float().clone())
        self.norm = nn.LayerNorm(hidden)
        self.project = nn.Linear(hidden, basis)
        self.mask_vec = nn.Parameter(emb[:self.vocab].float().mean(0))
        self.emb_rms = float(emb[:self.vocab].float().pow(2).mean().sqrt())
        if latent_mode == "hidden":
            self.thought = nn.Linear(hidden, hidden, bias=False)
            nn.init.eye_(self.thought.weight)
        self.last_parts = {}
        self._aux_detail = None

    @property
    def device(self):
        return self.basis.device

    def _embed(self, ids):
        return self.lm.get_input_embeddings()(ids)

    def _forward(self, embeds, position_ids, attention_mask, cache, keep):
        out = self.lm(inputs_embeds=embeds, position_ids=position_ids, attention_mask=attention_mask,
                      past_key_values=cache, use_cache=True, output_hidden_states=True, logits_to_keep=keep)
        return out.hidden_states[-1]  # after Qwen's final norm

    def prefix(self, prompt_ids, prompt_mask, stochastic=False):
        if self.prefix_adapter_off:
            with torch.no_grad(), self.lm.disable_adapter():
                return self._prefix(prompt_ids, prompt_mask, stochastic)
        return self._prefix(prompt_ids, prompt_mask, stochastic)

    def _prefix(self, prompt_ids, prompt_mask, stochastic=False):
        """Run prompt + R latent steps. Returns (cache, mask over prefix, next position per row, latent embeds, final hidden)."""
        cache, mask = DynamicCache(), prompt_mask
        n = prompt_mask.sum(-1)  # real prompt length; left padding means the last slot is always real
        h = self._forward(self._embed(prompt_ids), (prompt_mask.cumsum(-1) - 1).clamp(min=0),
                          prompt_mask, cache, 1)[:, -1]
        latents = []
        for r in range(self.steps):
            if self.latent_mode == "hidden":
                hf = h.float()
                e = (self.thought(hf / hf.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()) * self.emb_rms).to(h.dtype)[:, None]
                w = None
            else:
                logits = self.project(self.norm(h.float()))
                w = F.gumbel_softmax(logits, tau=self.tau) if stochastic else F.softmax(logits / self.tau, dim=-1)
                e = (w @ self.basis).to(h.dtype)[:, None]
            if self._trace is not None:
                self._trace.append((None if w is None else w.detach(), e.detach()))
            if self._latent_hook is not None:
                e = self._latent_hook(e)
            latents.append(e)
            mask = torch.cat([mask, mask.new_ones(len(mask), 1)], dim=1)
            h = self._forward(e, (n + r)[:, None], mask, cache, 1)[:, -1]
        return cache, mask, n + self.steps, latents, h

    def canvas_hidden(self, cache, prefix_mask, start, tokens):
        """Hidden states [B,A,D] for the answer canvas: sees the prefix and every canvas position."""
        b, a = tokens.shape
        emb = self._embed(tokens)
        emb = torch.where(tokens.eq(self.mask_id)[..., None], self.mask_vec.to(emb.dtype), emb)
        pm = prefix_mask.bool()
        if self.canvas_latents_only:
            pm = pm.clone(); pm[:, :-self.steps] = False
        visible = torch.cat([pm, pm.new_ones(b, a)], dim=1)[:, None, None, :]
        bias = torch.zeros(b, 1, a, visible.shape[-1], dtype=emb.dtype, device=emb.device)
        bias = bias.masked_fill(~visible, torch.finfo(emb.dtype).min)
        pos = start[:, None] + torch.arange(a, device=tokens.device)
        return self._forward(emb, pos, bias, cache, 1)[:, -a:]

    def aux_solution_loss(self, cache, pmask, start, h_last, sol, sol_mask):
        """Next-token loss on the written solution given ONLY the latent positions (prompt hidden from it)."""
        b, s = sol.shape
        plen, r = pmask.shape[1], self.steps
        head = self.lm.get_output_embeddings()
        # first token is predicted from the last latent position, the rest by teacher forcing over the latents
        first = head(h_last)[:, :self.vocab].float()
        ce = torch.zeros(b, s, device=sol.device)
        pred = torch.zeros(b, s, dtype=torch.long, device=sol.device)
        ce[:, 0] = F.cross_entropy(first, sol[:, 0], reduction="none")
        pred[:, 0] = first.argmax(-1)
        if s > 1:
            emb = self._embed(sol[:, :-1])
            allowed = torch.zeros(b, s - 1, plen + s - 1, dtype=torch.bool, device=sol.device)
            allowed[:, :, plen - r:plen] = True                      # latent positions only, never the prompt
            allowed[:, :, plen:] = torch.tril(torch.ones(s - 1, s - 1, dtype=torch.bool, device=sol.device))
            bias = torch.zeros(b, 1, s - 1, plen + s - 1, dtype=emb.dtype, device=emb.device)
            bias = bias.masked_fill(~allowed[:, None], torch.finfo(emb.dtype).min)
            pos = start[:, None] + torch.arange(s - 1, device=sol.device)
            h = self._forward(emb, pos, bias, cache, 1)
            cache.crop(plen)  # canvas must see the clean prefix
            sel = sol_mask[:, 1:].bool()
            lg = head(h[sel])[:, :self.vocab].float()
            ce[:, 1:][sel] = F.cross_entropy(lg, sol[:, 1:][sel], reduction="none")
            pred[:, 1:][sel] = lg.argmax(-1)
        if self._aux_detail is not None:   # diagnostics only (scripts/diagnose_latents.py)
            self._aux_detail.append(dict(ce=ce.detach(), pred=pred, target=sol, valid=sol_mask))
        return (ce * sol_mask).sum() / sol_mask.sum().clamp_min(1)

    def loss(self, prompt_ids, prompt_mask, target, full_mask_prob=0.75, stochastic=False, sol=None, sol_mask=None,
             aux_weight=0.0):
        """Masked-token cross-entropy on the answer canvas (no next-token shift: position i predicts token i),
        plus aux_weight * latent->solution reconstruction loss when a solution is given."""
        cache, pmask, start, _, h_last = self.prefix(prompt_ids, prompt_mask, stochastic)
        aux = None
        if sol is not None and aux_weight > 0 and self.steps > 0:
            aux = self.aux_solution_loss(cache, pmask, start, h_last, sol, sol_mask)
        noisy, masks, _ = corrupt(target, self.mask_id, self.pad_id, full_mask_prob=full_mask_prob)
        h = self.canvas_hidden(cache, pmask, start, noisy)
        logits = self.lm.get_output_embeddings()(h[masks])[:, :self.vocab].float()
        canvas = F.cross_entropy(logits, target[masks])
        self.last_parts = dict(canvas=canvas.item(), aux=None if aux is None else aux.item())
        return canvas if aux is None else canvas + aux_weight * aux

    @torch.no_grad()
    def generate(self, prompt_ids, prompt_mask, length, steps=8):
        """Greedy confidence-based iterative unmasking (same schedule as diffusion.sample), prefix cache reused."""
        cache, pmask, start, *_ = self.prefix(prompt_ids, prompt_mask)
        plen = pmask.shape[1]
        y = torch.full((len(prompt_ids), length), self.mask_id, dtype=torch.long, device=prompt_ids.device)
        for step in range(steps):
            h = self.canvas_hidden(cache, pmask, start, y)
            cache.crop(plen)  # drop the canvas KV; the prefix stays valid
            logits = self.lm.get_output_embeddings()(h)[..., :self.vocab].float()
            conf, pred = F.softmax(logits, dim=-1).max(-1)
            is_mask = y.eq(self.mask_id)
            candidate = torch.where(is_mask, pred, y)
            remaining = int(math.floor(length * (1 - (step + 1) / steps)))
            if step == steps - 1 or remaining == 0:
                y = candidate
            else:
                keep = torch.where(is_mask, conf, torch.ones_like(conf))
                y = candidate.scatter(1, keep.argsort(-1)[:, :remaining], self.mask_id)
        return y

    def trainable(self):
        return [p for p in self.parameters() if p.requires_grad]

    def extra_state(self):
        """Everything trainable that is not already in the LoRA adapter."""
        return {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")} | {
            k: v for k, v in self.state_dict().items() if k.startswith("lm.") and "lora_" in k}
