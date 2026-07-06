#include "pinn/core/backend.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


void cuda_add(const float *a, const float *b, float *out){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size){
        out[i] = a[i] + b[i];
    }
}