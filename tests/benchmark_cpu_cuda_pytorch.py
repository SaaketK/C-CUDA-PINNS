#!/usr/bin/env python3
"""Benchmark native CPU/CUDA autograd against PyTorch and save a text report."""
import pathlib, shlex, subprocess, time
import torch

ROOT=pathlib.Path(__file__).resolve().parents[2]
BUILD=pathlib.Path('/tmp/pinn-derivative-benchmark')
REPORT=ROOT/'src/tests/CPU-CUDA-PyTorch-derivative-results.txt'

def build():
    BUILD.mkdir(exist_ok=True); objs=[]
    cflags=shlex.split(subprocess.check_output(['pkg-config','--cflags','openblas'],text=True))
    libs=shlex.split(subprocess.check_output(['pkg-config','--libs','openblas'],text=True))
    for s in ['src/tests/derivative-benchmark.c','src/core/tensor.c','src/core/ops_cpu.c','src/core/ops.c','src/core/autograd.c','src/core/backend.c']:
        o=BUILD/(pathlib.Path(s).stem+'.o'); subprocess.run(['gcc','-std=c11','-O2','-DPINN_USE_CUDA','-DPINN_USE_OPENBLAS',*cflags,'-Iinclude','-fopenmp','-c',s,'-o',str(o)],cwd=ROOT,check=True); objs.append(str(o))
    cu=BUILD/'ops_cuda.o'; subprocess.run(['nvcc','-std=c++14','-O2','-Iinclude','-c','src/core/ops_cuda.cu','-o',str(cu)],cwd=ROOT,check=True)
    subprocess.run(['nvcc','-Xcompiler','-fopenmp',*objs,str(cu),'-lcublas',*libs,'-o',str(BUILD/'benchmark')],cwd=ROOT,check=True)

def torch_case(name):
    if name=='quadratic':
        x=torch.tensor([(i%23-11)/11 for i in range(2048)],dtype=torch.float32,requires_grad=True); loss=(x*x).mean(); ins=[x]
    elif name=='nonlinear':
        x=torch.tensor([(i%31-15)/10 for i in range(2048)],dtype=torch.float32,requires_grad=True); loss=torch.tanh(x*x+.25).mean(); ins=[x]
    else:
        x=torch.tensor([(i%29-14)/14 for i in range(2048)],dtype=torch.float32).reshape(64,32).requires_grad_(); w=torch.tensor([(i%17-8)/20 for i in range(512)],dtype=torch.float32).reshape(32,16).requires_grad_(); b=torch.tensor([(i-8)/30 for i in range(16)],dtype=torch.float32,requires_grad=True); target=torch.tensor([(i%13)/13 for i in range(1024)],dtype=torch.float32).reshape(64,16); loss=(torch.sigmoid(torch.tanh(x)@w+b)-target).square().mean(); ins=[x,w,b]
    loss.backward(); g=torch.cat([v.grad.flatten() for v in ins]); return loss.item(),g.sum().item(),g.square().sum().item()

def main():
    build(); out=subprocess.run([str(BUILD/'benchmark')],cwd=ROOT,text=True,capture_output=True,check=True).stdout
    native={};
    for line in out.splitlines():
        p=line.split(','); native[(p[1],p[2])]=tuple(map(float,p[3:]))
    lines=['CPU vs CUDA vs PyTorch derivative benchmark','', 'Times are end-to-end forward + backward milliseconds per iteration; lower is faster.', 'PyTorch is CPU-only in /tmp/pinn-torch-venv, so it is a correctness/performance CPU reference.','', 'Equation                 Backend       ms/iter    loss error   grad-sum error  grad-sqsum error']
    for eq in ['quadratic','nonlinear','dense']:
        ref=torch_case(eq)
        for backend in ['CPU','CUDA']:
            ms,loss,gs,gss=native[(eq,backend)]; lines.append(f'{eq:24} C {backend:5} {ms:10.4f} {abs(loss-ref[0]):11.3e} {abs(gs-ref[1]):14.3e} {abs(gss-ref[2]):16.3e}')
        start=time.perf_counter(); runs=100 if eq!='dense' else 30
        for _ in range(runs): torch_case(eq)
        lines.append(f'{eq:24} PyTorch CPU {(time.perf_counter()-start)*1000/runs:10.4f}       reference')
        lines.append('')
    REPORT.write_text('\n'.join(lines)+'\n'); print('\n'.join(lines)); print(f'\nSaved: {REPORT}')
if __name__=='__main__': main()
