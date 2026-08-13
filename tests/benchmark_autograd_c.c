#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "pinn/core/autograd.h"
#include "pinn/core/ops.h"
#include "pinn/core/tensor.h"
#include "pinn/autodiff/jet.h"
#include "pinn/nn/mlp.h"
#include "pinn/pinn/residual.h"
static double ms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1000.0+t.tv_nsec/1e6;}
static int cmp(const void*a,const void*b){float x=*(float*)a,y=*(float*)b;return(x>y)-(x<y);}
static void one(MLP*m,const float*x,int n){int s[2]={n,2};Tensor*p=tensor_from_data(x,s,2,0);Tape*t=tape_create();set_curr_tape(t);JetTape*j=jet_tape_create();set_curr_jet_tape(j);JetTensor*q=jet_create_input(p,2);JetTensor*y=jet_mlp_forward(m,q);Heat1DParams h={.alpha=.1f};Tensor*r=heat1d_ansatz_residual(y,p,&h);Tensor*l=residual_mse_loss(r);backward(l);jet_tape_free_shallow(j);jet_free(q);tape_free(t);}
int main(int argc,char**argv){int n=argc>1?atoi(argv[1]):1000,reps=argc>2?atoi(argv[2]):100;float*x=malloc(2*n*sizeof(float)),*v=malloc(reps*sizeof(float));for(int i=0;i<2*n;i++)x[i]=(float)i/(2*n-1);int sizes[]={2,64,64,64,1};srand(1234);MLP*m=mlp_create(sizes,5,NULL);for(int i=0;i<5;i++)one(m,x,n);for(int i=0;i<reps;i++){double a=ms();one(m,x,n);v[i]=(float)(ms()-a);}qsort(v,reps,sizeof(float),cmp);printf("C Jet CPU: median=%.3f ms p95=%.3f ms\n",v[reps/2],v[(int)(.95f*(reps-1))]);FILE*f=fopen("src/tests/benchmark_autograd_performance.txt","a");if(f){fprintf(f,"C Jet CPU batch=%d repeats=%d median=%.3f ms p95=%.3f ms\n",n,reps,v[reps/2],v[(int)(.95f*(reps-1))]);fclose(f);}mlp_free(m);free(x);free(v);}
