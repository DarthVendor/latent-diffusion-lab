"""Probe fixed-mask, two-dimensional local parameter-loss slices."""
import json
from pathlib import Path
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from latent_diffusion.models import LatentDiffusion, ModelConfig
from latent_diffusion.data import addition_data

def fixed_corrupt(y):
    return torch.full_like(y, 10), torch.ones_like(y, dtype=torch.bool), torch.ones(y.shape[0])

def loss(model, pair):
    x, y = pair
    noisy, mask, t = fixed_corrupt(y)
    out = model(x, noisy, t, stochastic=False)
    return F.cross_entropy(out[mask], y[mask]).item()

def make_directions(model, seed):
    gen = torch.Generator().manual_seed(seed)
    dirs = []
    for _ in range(2):
        direction = {}
        for n, p in model.named_parameters():
            rand = torch.randn(p.shape, generator=gen)
            wnorm = p.detach().norm().item()
            rnorm = rand.norm().item()
            direction[n] = rand * (wnorm / max(rnorm, 1e-12)) if wnorm > 1e-8 else torch.zeros_like(rand)
        dirs.append(direction)
    return dirs

def probe(model, pair, seed, coordinates):
    dirs = make_directions(model, seed)
    weights = {n: p.detach().clone() for n, p in model.named_parameters()}
    grid = []
    with torch.no_grad():
        for a in coordinates:
            row = []
            for b in coordinates:
                for n, p in model.named_parameters():
                    p.copy_(weights[n] + a * dirs[0][n] + b * dirs[1][n])
                row.append(loss(model, pair))
            grid.append(row)
        for n, p in model.named_parameters():
            p.copy_(weights[n])
    return np.asarray(grid)

def main():
    torch.set_num_threads(1)
    train, val = addition_data(seed=17)
    output = ROOT / 'results' / 'landscape'
    output.mkdir(parents=True, exist_ok=True)
    coords = np.linspace(-0.7, 0.7, 13)
    report = {'description': 'Two random filter-normalized parameter slices; all masked; deterministic inference.',
              'coordinates': coords.tolist(), 'results': {}}
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for c, name in enumerate(('baseline', 'latent_diffusion')):
        checkpoint = torch.load(ROOT / 'runs/validation200' / f'{name}_400.pt',
                                map_location='cpu', weights_only=True)
        model = LatentDiffusion(ModelConfig(**checkpoint['model_config']))
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        report['results'][name] = {}
        for r, (split, pair) in enumerate((('train', train), ('heldout', val))):
            grid = probe(model, pair, seed=3001, coordinates=coords)
            np.save(output / f'{name}_{split}.npy', grid)
            report['results'][name][split] = {
                'center_loss': float(grid[6, 6]), 'min_loss_on_slice': float(grid.min()),
                'max_loss_on_slice': float(grid.max()),
                'grid_file': f'{name}_{split}.npy', 'loss_grid': grid.tolist()
            }
            ax = axes[r, c]
            img = ax.contourf(coords, coords, np.log10(np.maximum(grid, 1e-7)), levels=18, cmap='viridis')
            ax.plot(0, 0, 'r+', markersize=10, markeredgewidth=2)
            ax.set(title=f'{name} | {split}', xlabel='Direction 2 scale', ylabel='Direction 1 scale')
            fig.colorbar(img, ax=ax, label='log10 cross-entropy')
        print(name, {k: round(v['center_loss'], 4) for k, v in report['results'][name].items()}, flush=True)
    (output / 'summary.json').write_text(json.dumps(report, indent=2))
    fig.savefig(output / 'landscape.png', dpi=155)

if __name__ == '__main__':
    main()
