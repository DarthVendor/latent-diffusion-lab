"""Resume toy checkpoints for 200 more steps with fresh AdamW states.

Original optimizer state was not saved; model weights resume, optimizer does not.
"""
import json
from pathlib import Path
import torch
from torch import nn
from latent_diffusion.data import addition_data
from latent_diffusion.diffusion import corrupt, masked_loss
from latent_diffusion.models import LatentDiffusion, ModelConfig
from latent_diffusion.train import accuracy, seed_all

root = Path(__file__).resolve().parents[1]
old = json.loads((root / 'results/validation200.json').read_text())
tc = old['configuration']['training']
seed_all(int(tc['seed']) + 200)
torch.set_num_threads(int(tc['threads']))
train, val = addition_data(seed=int(tc['split_seed']))
results = {'note': 'Resumed weights from step 200; AdamW optimizer restarted.',
           'training': {'additional_steps': 200, 'learning_rate': tc['lr'], 'seed': int(tc['seed'])+200},
           'models': {}}
for name in ['baseline', 'latent_diffusion']:
    checkpoint = torch.load(root / 'runs/validation200' / f'{name}.pt', map_location='cpu', weights_only=True)
    m = LatentDiffusion(ModelConfig(**checkpoint['model_config']))
    m.load_state_dict(checkpoint['state_dict'])
    optim = torch.optim.AdamW(m.parameters(), lr=float(tc['lr']), weight_decay=1e-4)
    history = []
    for step in range(201, 401):
        m.train()
        noisy, mask, t = corrupt(train[1], m.cfg.mask_id, m.cfg.pad_id)
        logits = m(train[0], noisy, t, stochastic=m.cfg.use_latent)
        loss = masked_loss(logits, train[1], mask)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        optim.step()
        if step % 25 == 0:
            tr, _ = accuracy(m, train, steps=int(tc['sampling_steps']))
            va, _ = accuracy(m, val, steps=int(tc['sampling_steps']))
            history.append({'step': step, 'loss': round(float(loss), 5), 'train_exact': tr, 'validation_exact': va})
            print(name, step, round(float(loss), 4), round(tr, 3), round(va, 3), flush=True)
    torch.save({'state_dict': m.state_dict(), 'model_config': checkpoint['model_config'],
                'optimizer_state_dict': optim.state_dict(), 'steps': 400},
               root / 'runs/validation200' / f'{name}_400.pt')
    results['models'][name] = {'history': history, 'final': history[-1],
                               'best_validation': max(history, key=lambda x: x['validation_exact'])}
(root / 'results/continuation400.json').write_text(json.dumps(results, indent=2))
