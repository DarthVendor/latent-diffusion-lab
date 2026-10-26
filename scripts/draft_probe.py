"""Cheap probe: can a masked-diffusion block drafter get enough tokens accepted per round to beat plain AR?
1. Qwen3-0.6B greedy completions (non-thinking, reasoning prompt) = the target's own output
2. train the block drafter (LoRA, no latent steps) to predict the next K target tokens from prompt + prefix
3. offline rollout: speculative rounds replayed against the target's fixed greedy text; report tokens/round
Tokens/round = accepted draft tokens + 1 (the target's own token from the verify pass).
Break-even vs compiled AR (~4 ms/token) for a same-size drafter costing ~2 passes/round: ~2.5 tokens/round.
Caveat: greedy text was generated in padded batches, so it can differ from batch-1 greedy by numerical noise.
"""
import argparse, json, random, sys, time
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "scripts")
from train_qwen import prompt_text  # noqa: E402
from latent_diffusion.qwen_latent import QwenLatentDiffusion  # noqa: E402

SUFFIX = "\nThink step by step, then finish with 'The final answer is N'."
EOS = {151645, 151643}  # <|im_end|>, <|endoftext|>


def left_pad(seqs, pad_id):
    n = max(map(len, seqs))
    ids = torch.full((len(seqs), n), pad_id)
    am = torch.zeros(len(seqs), n, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, n - len(s):], am[i, n - len(s):] = torch.tensor(s), 1
    return ids, am


@torch.no_grad()
def gen_data(tok, questions, bs=64, new=384):
    lm = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16).cuda().eval()
    order = sorted(range(len(questions)), key=lambda i: len(questions[i]))  # similar lengths per batch = less padding
    out = [None] * len(questions)
    for b in range(0, len(order), bs):
        idx = order[b:b + bs]
        enc = tok([prompt_text(tok, questions[i], SUFFIX) for i in idx], return_tensors="pt", padding=True).to("cuda")
        gen = lm.generate(**enc, max_new_tokens=new, do_sample=False)[:, enc.input_ids.shape[1]:].tolist()
        for i, row, am in zip(idx, gen, enc.attention_mask.cpu()):
            cut = next((j + 1 for j, t in enumerate(row) if t in EOS), len(row))
            out[i] = dict(prompt=enc.input_ids[idx.index(i)][am.bool()].tolist(), comp=row[:cut])
        print(f"generated {b + len(idx)}/{len(order)}", flush=True)
    del lm; torch.cuda.empty_cache()
    return out


def make_batch(data, picks, K, end_id, pad_id):
    ctxs, tgts = [], []
    for ex, t in picks:
        d = data[ex]
        ctxs.append(d["prompt"] + d["comp"][:t])
        tg = d["comp"][t:t + K]
        tgts.append(tg + [end_id] * (K - len(tg)))
    return (*left_pad(ctxs, pad_id), torch.tensor(tgts))


@torch.no_grad()
def rollout(m, data, K, steps, bs=32):
    """Replay speculative decoding against each example's fixed greedy text (what the verify pass would produce)."""
    m.eval()
    pos, rounds = [0] * len(data), [0] * len(data)
    hit = [0] * K; tried = [0] * K  # P(match at draft position j | positions < j matched)
    active = list(range(len(data)))
    while active:
        for c in range(0, len(active), bs):
            chunk = active[c:c + bs]
            ids, am = left_pad([data[i]["prompt"] + data[i]["comp"][:pos[i]] for i in chunk], m.pad_id)
            drafts = m.generate(ids.cuda(), am.cuda(), K, steps).tolist()
            for i, d in zip(chunk, drafts):
                rem = data[i]["comp"][pos[i]:pos[i] + K]
                a = 0
                while a < len(rem) and d[a] == rem[a]:
                    hit[a] += 1; tried[a] += 1; a += 1
                if a < len(rem):
                    tried[a] += 1
                pos[i] += min(a + 1, len(data[i]["comp"]) - pos[i]); rounds[i] += 1
        active = [i for i in active if pos[i] < len(data[i]["comp"])]
    tokens, nr = sum(len(d["comp"]) for d in data), sum(rounds)
    return dict(tokens_per_round=tokens / nr, rounds=nr, tokens=tokens,
                cond_accept_by_position=[h / t if t else None for h, t in zip(hit, tried)])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=3000); ap.add_argument("--n-eval", type=int, default=100)
    ap.add_argument("--K", type=int, default=8); ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="draft"); ap.add_argument("--data", default="runs/draft/data.pt")
    ap.add_argument("--shared-cache", action="store_true", help="drafter reads the target's (adapter-off) cache")
    a = ap.parse_args()
    random.seed(0); torch.manual_seed(0)
    root = Path("runs") / a.name; root.mkdir(parents=True, exist_ok=True); Path("results").mkdir(exist_ok=True)
    ds = load_dataset("openai/gsm8k", "main")
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")

    if Path(a.data).exists():
        train, evald = torch.load(a.data)
    else:
        t0 = time.time()
        qs = list(ds["train"]["question"][:a.n_train]) + list(ds["test"]["question"][:a.n_eval])
        both = gen_data(tok, qs)
        train, evald = both[:a.n_train], both[a.n_train:]
        torch.save((train, evald), root / "data.pt"); print(f"data generated in {time.time() - t0:.0f}s", flush=True)
    avg = sum(len(d["comp"]) for d in train) / len(train)
    print(f"train {len(train)} completions, avg {avg:.0f} tokens; eval {len(evald)}", flush=True)

    m = QwenLatentDiffusion(steps=0, prefix_adapter_off=a.shared_cache).cuda()  # no latent steps: pure block drafter
    end_id = tok.convert_tokens_to_ids("<|im_end|>")
    opt = torch.optim.AdamW(m.trainable(), lr=a.lr, weight_decay=0.0)
    hist = []
    for step in range(1, a.steps + 1):
        m.train()
        picks = [(e := random.randrange(len(train)), random.randrange(len(train[e]["comp"]))) for _ in range(a.bs)]
        ids, am, tgt = make_batch(train, picks, a.K, end_id, m.pad_id)
        loss = m.loss(ids.cuda(), am.cuda(), tgt.cuda())
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.trainable(), 1.0); opt.step()
        if step % 10 == 0:
            print(f"step {step} loss {loss.item():.3f} peak {torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
        if step % 500 == 0 or step == a.steps:
            r = rollout(m, evald[:30], a.K, 1)
            hist.append(dict(step=step, loss=loss.item(), tokens_per_round_1step_30ex=r["tokens_per_round"]))
            print(hist[-1], flush=True)

    res = dict(args=vars(a), avg_completion_tokens=avg, history=hist, rollout={})
    for s in (1, 2, 4):
        t0 = time.time(); res["rollout"][f"denoise_steps_{s}"] = r = rollout(m, evald, a.K, s)
        r["seconds"] = time.time() - t0; print(s, json.dumps(r), flush=True)
    Path(f"results/draft_probe_{a.name}.json").write_text(json.dumps(res, indent=1))
    torch.save(m.extra_state(), root / "drafter.pt")
