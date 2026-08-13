#include "pinn/nn/optimizer_cuda.h"
#include <cuda_runtime.h>
#include <math.h>

static __global__ void sgd_step_kernel(float *data, const float *grad, float lr, int size){ 
    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < size) data[i] -= lr * grad[i]; 
}

extern "C" int cuda_sgd_step(float *data, const float *grad, float lr, int size){ 
    if(!data || !grad || size < 0) return 1; 
    if(size == 0 || lr == 0.0f) return 0; 
    int threads_per_block = 256; 
    int blocks = (size + threads_per_block - 1) / threads_per_block; 
    sgd_step_kernel<<<blocks, threads_per_block>>>(data, grad, lr, size); 
    return cudaGetLastError() == cudaSuccess ? 0 : 1; 
}

static __global__ void adam_step_kernel(float *data, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float eps, float beta1_correction, 
float beta2_correction, int size){ 

    int i = blockIdx.x * blockDim.x + threadIdx.x; 
    if(i < size){ 
        float g = grad[i]; 
        m[i] = beta1 * m[i] + (1.0f - beta1) * g; 
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g; 
        data[i] -= lr * (m[i] / beta1_correction) / (sqrtf(v[i] / beta2_correction) + eps); 
    } 
}

extern "C" int cuda_adam_step(float *data, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float eps, float beta1_correction, 
float beta2_correction, int size){ 
    
    if(!data || !grad || !m || !v || size < 0) return 1; 
    if(size == 0) return 0; int threads_per_block = 256; 
    int blocks = (size + threads_per_block - 1) / threads_per_block; 
    adam_step_kernel<<<blocks, threads_per_block>>>(data, grad, m, v, lr, beta1, beta2, eps, beta1_correction, beta2_correction, size); 
    return cudaGetLastError() == cudaSuccess ? 0 : 1; 
}
