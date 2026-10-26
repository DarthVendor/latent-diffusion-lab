"""Train and compare latent vs no-latent configurations."""
import json
import random
from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn
from .data import addition_data
from .diffusion import corrupt, masked_loss, sample
from .models import LatentDiffusion, ModelConfig

def seed_all(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def accuracy(model, pair, steps=8):
    model.eval()
    x, target = pair
    pred = sample(model, x, target.shape[1], steps=steps)
    return float(pred.eq(target).all(-1).float().mean()), pred

def run(cfg):
    run_cfg = cfg['training']
    seed_all(int(run_cfg['seed']))
    device = torch.device(run_cfg.get('device', 'cpu'))
    torch.set_num_threads(int(run_cfg.get('threads', 1)))
    root = Path(run_cfg.get('output_dir', 'runs/addition'))
    root.mkdir(parents=True, exist_ok=True)
    train_pair, val_pair = addition_data(seed=int(run_cfg.get('split_seed', 17)))
    train_pair = tuple(v.to(device) for v in train_pair)
    val_pair = tuple(v.to(device) for v in val_pair)
    results = {'configuration': cfg, 'models': {}}
    for name, active in [('baseline', False), ('latent_diffusion', True)]:
        seed_all(int(run_cfg['seed']))
        mc = ModelConfig(**{**cfg['model'], 'use_latent': active})
        model = LatentDiffusion(mc).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=float(run_cfg['lr']), weight_decay=1e-4)
        history = []
        iterations = int(run_cfg['iterations'])
        for iteration in range(1, iterations + 1):
            model.train()
            x, target = train_pair
            noisy, mask, t = corrupt(target, mc.mask_id, mc.pad_id)
            logits = model(x, noisy, t, stochastic=active)
            loss = masked_loss(logits, target, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if iteration == 1 or iteration % int(run_cfg['eval_every']) == 0 or iteration == iterations:
                tr, _ = accuracy(model, train_pair, steps=int(run_cfg['sampling_steps']))
                va, _ = accuracy(model, val_pair, steps=int(run_cfg['sampling_steps']))
                history.append({'step': iteration, 'loss': float(loss.item()), 'train_exact': tr, 'validation_exact': va})
                print(f'{name:18} step={iteration:4} loss={loss:.3f} train={tr:.3f} val={va:.3f}', flush=True)
        train_acc, _ = accuracy(model, train_pair, steps=int(run_cfg['sampling_steps']))
        val_acc, pred = accuracy(model, val_pair, steps=int(run_cfg['sampling_steps']))
        results['models'][name] = {
            'train_exact': train_acc, 'validation_exact': val_acc,
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'history': history,
            'examples': [{'input': a, 'target': b, 'prediction': c}
                         for a, b, c in zip(val_pair[0][:10].tolist(), val_pair[1][:10].tolist(), pred[:10].tolist())]
        }
        torch.save({'state_dict': model.state_dict(), 'model_config': asdict(mc)}, root / f'{name}.pt')
    (root / 'results.json').write_text(json.dumps(results, indent=2))
    return results
