#!/usr/bin/env python3
import argparse, csv, json, time
from pathlib import Path
import torch

class HeatMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.ModuleList([torch.nn.Linear(2,64), torch.nn.Linear(64,64), torch.nn.Linear(64,64), torch.nn.Linear(64,1)])
    def forward(self, x):
        for layer in self.layers[:-1]: x = torch.tanh(layer(x))
        return self.layers[-1](x)

def residual(model, points):
    points = points.detach().requires_grad_(True); raw = model(points); t, x = points[:,:1], points[:,1:]
    u = (1-t)*torch.sin(torch.pi*x) + t*x*(1-x)*raw
    u_t, u_x = torch.autograd.grad(u, points, torch.ones_like(u), create_graph=True)[0].split(1, dim=1)
    u_xx = torch.autograd.grad(u_x, points, torch.ones_like(u_x), create_graph=True)[0][:,1:]
    return u_t - .1*u_xx

def main():
    p=argparse.ArgumentParser(); p.add_argument('--steps',type=int,default=10); p.add_argument('--points',type=int,default=128); p.add_argument('--seed',type=int,default=1234); p.add_argument('--device',choices=('cpu','cuda'),default='cuda'); p.add_argument('--threads',type=int,default=16); p.add_argument('--results-dir',default='src/tests/results/heat1d'); a=p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    d=torch.device(a.device); torch.manual_seed(a.seed); torch.cuda.manual_seed_all(a.seed) if d.type=='cuda' else None
    model=HeatMLP().to(d); opt=torch.optim.Adam(model.parameters(),lr=1e-3,betas=(.9,.999),eps=1e-8); points=torch.rand((a.points,2),generator=torch.Generator(device=d).manual_seed(a.seed),device=d); losses=[]
    if d.type=='cuda': torch.cuda.synchronize()
    start=time.perf_counter()
    for step in range(a.steps):
        opt.zero_grad(set_to_none=True); loss=residual(model,points).square().mean(); loss.backward(); opt.step(); losses.append(loss.item())
        if step in (0,a.steps-1): print(f'step={step} physics={loss.item():.6f} total={loss.item():.6f}')
    if d.type=='cuda': torch.cuda.synchronize()
    elapsed=time.perf_counter()-start; out=Path(a.results_dir); out.mkdir(parents=True,exist_ok=True); label=f'pytorch_{d.type}'
    with (out/f'{label}_loss.csv').open('w',newline='') as f: w=csv.writer(f); w.writerow(('step','physics_loss')); w.writerows(enumerate(losses))
    (out/f'{label}_metrics.json').write_text(json.dumps({'implementation':'PyTorch','device':d.type.upper(),'steps':a.steps,'collocation_points':a.points,'seed':a.seed,'dtype':'float32','training_seconds':elapsed,'milliseconds_per_step':elapsed*1000/a.steps,'initial_loss':losses[0],'final_loss':losses[-1]},indent=2)+'\n')
    print(f'training seconds={elapsed:.6f}')
if __name__=='__main__': main()
