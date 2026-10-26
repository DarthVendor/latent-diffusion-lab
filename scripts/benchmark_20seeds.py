"""Multi-seed, fixed-holdout diffusion vs autoregressive benchmark."""
import argparse, json, random, sys, time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from latent_diffusion.models import ModelConfig, LatentDiffusion, LatentReasoner, MemoryCompressor
from latent_diffusion.data import addition_data
from latent_diffusion.diffusion import corrupt, masked_loss, sample

class ARDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        self.embed=nn.Embedding(cfg.vocab_size,cfg.d_model)
        self.pos=nn.Embedding(cfg.max_answer_length,cfg.d_model)
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.n_heads,4*cfg.d_model,dropout=0.,batch_first=True,activation='gelu')
        self.blocks=nn.TransformerEncoder(layer,num_layers=cfg.decoder_layers,enable_nested_tensor=False)
        self.head=nn.Linear(cfg.d_model,cfg.vocab_size)
    def forward(self, memory, previous):
        b,n=previous.shape; m=memory.shape[1]
        seq=torch.cat([memory,self.embed(previous)+self.pos(torch.arange(n,device=previous.device))],dim=1)
        # memory positions must not look ahead to answer; answer sees memory + prefix
        mask=torch.zeros((m+n,m+n),device=seq.device,dtype=torch.bool)
        mask[:m,m:]=True
        mask[m:,m:]=torch.ones((n,n),device=seq.device,dtype=torch.bool).triu(1)
        return self.head(self.blocks(seq,mask=mask)[:,m:])
class ARModel(nn.Module):
    def __init__(self,cfg):
        super().__init__();self.cfg=cfg
        self.reasoner=LatentReasoner(cfg)
        self.compressor=MemoryCompressor(cfg)
        self.decoder=ARDecoder(cfg)
    def encode(self,prompt):return self.compressor(self.reasoner(prompt,False))
    def forward(self,prompt,target):
        previous=torch.cat([torch.full_like(target[:,:1],12),target[:,:-1]],dim=1)
        return self.decoder(self.encode(prompt),previous)
    @torch.no_grad()
    def generate(self,prompt,n):
        mem=self.encode(prompt)
        previous=torch.full((len(prompt),1),12,dtype=torch.long,device=prompt.device)
        out=[]
        for _ in range(n):
            logits=self.decoder(mem,previous)[:,-1,:10]
            token=logits.argmax(-1,keepdim=True)
            out.append(token)
            previous=torch.cat([previous,token],dim=1)
        return torch.cat(out,1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,default=20)
    ap.add_argument('--steps',type=int,default=250)
    ap.add_argument('--lr',type=float,default=3e-4)
    ap.add_argument('--size',type=int,default=64)
    ap.add_argument('--output',default='results/benchmark_20seeds.json')
    args=ap.parse_args();torch.set_num_threads(1)
    train,val=addition_data(seed=17)
    cfg=ModelConfig(d_model=args.size,n_heads=4,decoder_layers=2,latent_basis=args.size*2,reasoning_steps=4,memory_slots=4)
    records=[];start=time.perf_counter()
    for seed in range(args.seeds):
        for name in ('diffusion','latent_diffusion','ar','latent_ar'):
            torch.manual_seed(1000+seed);random.seed(1000+seed)
            is_latent=name.startswith('latent_')
            cfg.use_latent=is_latent
            model=ARModel(cfg) if name.endswith('ar') else LatentDiffusion(cfg)
            opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
            for step in range(args.steps):
                model.train()
                if name.endswith('ar'):
                    logits=model(train[0],train[1]);loss=F.cross_entropy(logits[:,:,:10].transpose(1,2),train[1])
                else:
                    noisy,mask,t=corrupt(train[1],cfg.mask_id,cfg.pad_id,full_mask_prob=.75)
                    logits=model(train[0],noisy,t,stochastic=False)
                    loss=masked_loss(logits,train[1],mask)
                opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
            model.eval()
            with torch.no_grad():
                if name.endswith('ar'):
                    pred=model.generate(val[0],2);train_pred=model.generate(train[0],2)
                else:
                    pred=sample(model,val[0],2,steps=4);train_pred=sample(model,train[0],2,steps=4)
            rec={'seed':seed,'model':name,'validation_exact':float(pred.eq(val[1]).all(-1).float().mean()),'train_exact':float(train_pred.eq(train[1]).all(-1).float().mean()),'parameters':sum(p.numel() for p in model.parameters()),'train_loss_last':float(loss.detach())}
            records.append(rec)
            print('seed',seed,name,'val',round(rec['validation_exact'],3),'train',round(rec['train_exact'],3),'elapsed',round(time.perf_counter()-start,1),flush=True)
    grouped={}
    for name in ['diffusion','latent_diffusion','ar','latent_ar']:
        x=torch.tensor([v['validation_exact'] for v in records if v['model']==name])
        grouped[name]={'mean':float(x.mean()),'std':float(x.std(unbiased=True)),'min':float(x.min()),'max':float(x.max()),'median':float(x.median()),'parameters':next(v['parameters'] for v in records if v['model']==name)}
    result={'setup':vars(args),'split_seed':17,'train_size':80,'validation_size':20,'seconds':time.perf_counter()-start,'summary':grouped,'records':records}
    output=ROOT/args.output;output.parent.mkdir(exist_ok=True,parents=True);output.write_text(json.dumps(result,indent=2))
    print('SUMMARY',json.dumps(grouped,indent=2),flush=True)
if __name__=='__main__':main()