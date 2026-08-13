#!/usr/bin/env python3
"""Raw PyTorch autograd timings: no sampling or optimizer update."""
import argparse, statistics, time
import torch

def run(device, batch, repeats, report):
    torch.set_num_threads(16); torch.set_num_interop_threads(1)
    torch.manual_seed(1234); d=torch.device(device)
    layers=[torch.nn.Linear(2,64),torch.nn.Linear(64,64),torch.nn.Linear(64,64),torch.nn.Linear(64,1)]
    model=torch.nn.Sequential(*[x.to(d) for x in layers])
    points=torch.linspace(0,1,batch*2,device=d).reshape(batch,2)
    times=[]
    for iteration in range(repeats+5):
        if d.type=='cuda': torch.cuda.synchronize()
        start=time.perf_counter(); x=points.detach().requires_grad_(True); y=x
        for layer in layers[:-1]: y=torch.tanh(layer(y))
        y=layers[-1](y); t,q=x[:,:1],x[:,1:]; u=(1-t)*torch.sin(torch.pi*q)+t*q*(1-q)*y
        g=torch.autograd.grad(u,x,torch.ones_like(u),create_graph=True)[0]; uxx=torch.autograd.grad(g[:,1:2],x,torch.ones_like(g[:,:1]),create_graph=True)[0][:,1:2]
        loss=(g[:,:1]-.1*uxx).square().mean(); loss.backward()
        if d.type=='cuda': torch.cuda.synchronize()
        if iteration>=5: times.append((time.perf_counter()-start)*1000)
        model.zero_grad(set_to_none=True)
    median=statistics.median(times); p95=sorted(times)[int(.95*(len(times)-1))]
    result=f'PyTorch {device} batch={batch} repeats={repeats} median={median:.3f} ms p95={p95:.3f} ms'
    print(result)
    with open(report, 'a', encoding='utf-8') as output:
        output.write(result + '\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--device',default='cuda');p.add_argument('--batch',type=int,default=1000);p.add_argument('--repeats',type=int,default=100);p.add_argument('--report',default='src/tests/benchmark_autograd_performance.txt');a=p.parse_args();run(a.device,a.batch,a.repeats,a.report)
