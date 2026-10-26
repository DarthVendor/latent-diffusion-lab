"""Long-horizon matched-seed recurrent latent diffusion vs recurrent latent AR."""
import argparse,json,random,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from benchmark_20seeds import ARModel
from latent_diffusion.models import ModelConfig, LatentDiffusion
from latent_diffusion.data import addition_data
from latent_diffusion.diffusion import corrupt,masked_loss,sample

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--seeds',type=int,default=20)
 p.add_argument('--steps',type=int,default=250)
 p.add_argument('--size',type=int,default=64)
 p.add_argument('--lr',type=float,default=3e-4)
 p.add_argument('--output',default='results/latent_long.json')
 a=p.parse_args();torch.set_num_threads(1)
 train,val=addition_data(seed=17)
 cfg=ModelConfig(d_model=a.size,n_heads=4,decoder_layers=2,latent_basis=a.size*2,reasoning_steps=4,memory_slots=4,use_latent=True)
 records=[];begin=time.perf_counter()
 for seed in range(a.seeds):
  for name in ('latent_diffusion','latent_ar'):
   torch.manual_seed(1000+seed);random.seed(1000+seed)
   model=ARModel(cfg) if name=='latent_ar' else LatentDiffusion(cfg)
   opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
   history=[]
   for step in range(1,a.steps+1):
    model.train()
    if name=='latent_ar':
     logits=model(train[0],train[1]);loss=F.cross_entropy(logits[:,:,:10].transpose(1,2),train[1])
    else:
     noisy,mask,t=corrupt(train[1],cfg.mask_id,cfg.pad_id,full_mask_prob=.75)
     logits=model(train[0],noisy,t,stochastic=False);loss=masked_loss(logits,train[1],mask)
    opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
    if step%50==0 or step==a.steps:
     model.eval()
     with torch.no_grad():
      pred=model.generate(val[0],2) if name=='latent_ar' else sample(model,val[0],2,steps=4)
      tp=model.generate(train[0],2) if name=='latent_ar' else sample(model,train[0],2,steps=4)
     history.append({'step':step,'validation_exact':float(pred.eq(val[1]).all(-1).float().mean()),'train_exact':float(tp.eq(train[1]).all(-1).float().mean()),'loss':float(loss)})
   records.append({'seed':seed,'model':name,'history':history,'parameters':sum(p.numel() for p in model.parameters())})
   print('SEED',seed,name,'FINAL',history[-1],'ELAPSED',round(time.perf_counter()-begin,1),flush=True)
  out=ROOT/a.output;out.parent.mkdir(parents=True,exist_ok=True)
  out.write_text(json.dumps({'setup':vars(a),'completed_seeds':seed+1,'records':records},indent=2))
 print('DONE',round(time.perf_counter()-begin,1))
if __name__=='__main__':main()