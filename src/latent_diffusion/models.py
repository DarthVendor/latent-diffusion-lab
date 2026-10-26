"""Independent SLT-style latent projector and masked diffusion decoder."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class ModelConfig:
    vocab_size: int = 14
    mask_id: int = 10
    pad_id: int = 11
    d_model: int = 64
    n_heads: int = 4
    decoder_layers: int = 2
    reasoning_steps: int = 4
    latent_basis: int = 128
    temperature: float = 0.7
    max_answer_length: int = 8
    dropout: float = 0.0
    use_latent: bool = True
    memory_slots: int = 4

class LatentReasoner(nn.Module):
    """Small GRU toy backbone with continuous Gumbel-softmax reasoning updates."""
    def __init__(self, cfg):
        super().__init__()
        d = cfg.d_model
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, d)
        self.prompt_position = nn.Embedding(16, d)
        self.prompt_encoder = nn.GRU(d, d, batch_first=True)
        self.norm = nn.LayerNorm(d)
        self.project = nn.Linear(d, cfg.latent_basis)
        self.basis = nn.Parameter(torch.randn(cfg.latent_basis, d) * 0.12)
        self.transition = nn.GRUCell(d, d)

    def forward(self, prompt, stochastic=False):
        if prompt.shape[1] > 16:
            raise ValueError('toy prompt exceeds max length 16')
        pos = self.prompt_position(torch.arange(prompt.shape[1], device=prompt.device))
        _, state = self.prompt_encoder(self.embed(prompt) + pos)
        state = state.squeeze(0)
        states = [state]
        if self.cfg.use_latent:
            for _ in range(self.cfg.reasoning_steps):
                logits = self.project(self.norm(state))
                weights = (F.gumbel_softmax(logits, tau=self.cfg.temperature)
                           if stochastic else F.softmax(logits / self.cfg.temperature, dim=-1))
                state = self.transition(weights @ self.basis, state)
                states.append(state)
        return torch.stack(states, dim=1)

class MemoryCompressor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(cfg.memory_slots, cfg.d_model) * 0.02)
        self.attn = nn.MultiheadAttention(cfg.d_model, cfg.n_heads, batch_first=True)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, states):
        q = self.queries.unsqueeze(0).expand(states.shape[0], -1, -1)
        result, _ = self.attn(q, states, states, need_weights=False)
        return self.norm(result + q)

class MaskedDiffusionDecoder(nn.Module):
    """Bidirectional masked-token predictor conditioned on reasoning memory."""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.position = nn.Embedding(cfg.max_answer_length, cfg.d_model)
        self.timestep = nn.Sequential(nn.Linear(1, cfg.d_model), nn.SiLU(), nn.Linear(cfg.d_model, cfg.d_model))
        block = nn.TransformerEncoderLayer(cfg.d_model, cfg.n_heads, cfg.d_model * 4,
                                           dropout=cfg.dropout, batch_first=True, activation='gelu')
        self.blocks = nn.TransformerEncoder(block, num_layers=cfg.decoder_layers, enable_nested_tensor=False)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, noisy, t, memory):
        b, n = noisy.shape
        if n > self.cfg.max_answer_length:
            raise ValueError('target exceeds configured max_answer_length')
        p = self.position(torch.arange(n, device=noisy.device))
        t_emb = self.timestep(t.reshape(b, 1).to(dtype=memory.dtype)).unsqueeze(1)
        output = self.embed(noisy) + p + t_emb
        combined = torch.cat([memory, output], dim=1)
        return self.head(self.blocks(combined)[:, -n:])

class LatentDiffusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.reasoner = LatentReasoner(cfg)
        self.compressor = MemoryCompressor(cfg)
        self.decoder = MaskedDiffusionDecoder(cfg)

    def encode(self, prompt, stochastic=False):
        return self.compressor(self.reasoner(prompt, stochastic))

    def forward(self, prompt, noisy, t, stochastic=False):
        return self.decoder(noisy, t, self.encode(prompt, stochastic))

class QwenReasoner(nn.Module):
    """Experimental adapter; not connected to the CLI training pipeline."""
    def __init__(self, model_name, decoder_dim, basis=512, reasoning_steps=4, freeze_backbone=True):
        super().__init__()
        try:
            from transformers import AutoModel
        except ImportError as exc:
            raise ImportError('Install with: pip install -e ".[qwen]"') from exc
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden = self.backbone.config.hidden_size
        self.norm = nn.LayerNorm(hidden)
        self.project = nn.Linear(hidden, basis)
        self.basis = nn.Parameter(torch.randn(basis, hidden) * 0.02)
        self.transition = nn.GRUCell(hidden, hidden)
        self.to_decoder = nn.Linear(hidden, decoder_dim)
        self.steps = reasoning_steps
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, input_ids, attention_mask=None, stochastic=False):
        with torch.set_grad_enabled(not self.freeze_backbone):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        if attention_mask is not None:
            idx = attention_mask.long().sum(-1).clamp(min=1) - 1
            state = outputs.last_hidden_state[torch.arange(len(input_ids), device=input_ids.device), idx]
        else:
            state = outputs.last_hidden_state[:, -1]
        states = [state]
        for _ in range(self.steps):
            logits = self.project(self.norm(state))
            weights = F.gumbel_softmax(logits, tau=0.7) if stochastic else F.softmax(logits / 0.7, dim=-1)
            state = self.transition(weights @ self.basis, state)
            states.append(state)
        return self.to_decoder(torch.stack(states, dim=1))
