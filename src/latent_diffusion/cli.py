"""Command-line entrypoint."""
import argparse
from pathlib import Path
import yaml
from .train import run

def main():
    parser = argparse.ArgumentParser(description='Latent reasoning + masked diffusion research lab')
    parser.add_argument('command', choices=['train'])
    parser.add_argument('--config', default='configs/addition.yaml')
    parser.add_argument('--iters', type=int, default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    if args.iters is not None:
        config['training']['iterations'] = args.iters
    if args.out is not None:
        config['training']['output_dir'] = args.out
    run(config)

if __name__ == '__main__':
    main()
