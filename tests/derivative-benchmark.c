#define _POSIX_C_SOURCE 199309L
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include "pinn/core/autograd.h"
#include "pinn/core/backend.h"
#include "pinn/core/ops.h"
#include "pinn/core/tensor.h"

typedef struct { float loss, sum, sqsum; } Result;
typedef void (*Equation)(Device, Result *);

static double now_ms(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1000.0 + t.tv_nsec / 1e6; }
static void stats(Result *r, Tensor *loss, Tensor *a, Tensor *b, Tensor *c){
    Tensor *inputs[3] = {a, b, c}; r->sum = r->sqsum = 0.0f; tensor_to_cpu(loss); r->loss = loss->data[0];
    for(int j = 0; j < 3; j++) if(inputs[j]){ tensor_to_cpu(inputs[j]); for(int i = 0; i < inputs[j]->size; i++){ r->sum += inputs[j]->grad[i]; r->sqsum += inputs[j]->grad[i] * inputs[j]->grad[i]; } }
}
static void quadratic(Device d, Result *r){
    int sh[1]={2048}; float v[2048]; for(int i=0;i<2048;i++) v[i]=(float)(i%23-11)/11.0f;
    Tensor *x=tensor_from_data_device(v,sh,1,1,d); Tape *t=tape_create(); set_curr_tape(t); Tensor *l=tensor_mean(tensor_square(x)); backward(l); backend_sync(d); stats(r,l,x,NULL,NULL); tape_free(t); tensor_free(x);
}
static void nonlinear(Device d, Result *r){
    int sh[1]={2048}; float v[2048]; for(int i=0;i<2048;i++) v[i]=(float)(i%31-15)/10.0f;
    Tensor *x=tensor_from_data_device(v,sh,1,1,d); Tape *t=tape_create(); set_curr_tape(t); Tensor *a=tensor_scalar_add(tensor_square(x),0.25f); Tensor *l=tensor_mean(tensor_tanh(a)); backward(l); backend_sync(d); stats(r,l,x,NULL,NULL); tape_free(t); tensor_free(x);
}
static void dense(Device d, Result *r){
    int xs[2]={64,32}, ws[2]={32,16}, bs[1]={16}, ts[2]={64,16}; float x[2048],w[512],b[16],target[1024];
    for(int i=0;i<2048;i++)x[i]=(float)(i%29-14)/14.0f; for(int i=0;i<512;i++)w[i]=(float)(i%17-8)/20.0f; for(int i=0;i<16;i++)b[i]=(float)(i-8)/30.0f; for(int i=0;i<1024;i++)target[i]=(float)(i%13)/13.0f;
    Tensor *a=tensor_from_data_device(x,xs,2,1,d),*q=tensor_from_data_device(w,ws,2,1,d),*z=tensor_from_data_device(b,bs,1,1,d),*y=tensor_from_data_device(target,ts,2,0,d);
    Tape *t=tape_create(); set_curr_tape(t); Tensor *h=tensor_tanh(a); Tensor *p=tensor_sigmoid(tensor_bias_add(tensor_matmult(h,q),z)); Tensor *l=tensor_mse(p,y); backward(l); backend_sync(d); stats(r,l,a,q,z); tape_free(t); tensor_free(y); tensor_free(z); tensor_free(q); tensor_free(a);
}
static void bench(const char *name, Equation fn, Device d, int runs){ Result r; for(int i=0;i<3;i++)fn(d,&r); double start=now_ms(); for(int i=0;i<runs;i++)fn(d,&r); double elapsed=(now_ms()-start)/runs; printf("RESULT,%s,%s,%.6f,%.9g,%.9g,%.9g\n",name,d==DEVICE_CUDA?"CUDA":"CPU",elapsed,r.loss,r.sum,r.sqsum); }
int main(void){ bench("quadratic",quadratic,DEVICE_CPU,100); bench("nonlinear",nonlinear,DEVICE_CPU,100); bench("dense",dense,DEVICE_CPU,30); if(backend_cuda_available()){ bench("quadratic",quadratic,DEVICE_CUDA,100); bench("nonlinear",nonlinear,DEVICE_CUDA,100); bench("dense",dense,DEVICE_CUDA,30); } return 0; }
