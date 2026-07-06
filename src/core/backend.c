#include "pinn/core/backend.h"

#include <stdio.h>

#ifdef PINN_USE_CUDA
    #include <cuda_runtime.h>
#endif

static int g_cuda_available = 0;
static Device g_default_device = DEVICE_CPU;

void backend_init(void){
#ifdef PINN_USE_CUDA
    int count = 0;
    cudaError_t err = cudaGetDeviceCount(&count);
    g_cuda_available = (err == cudaSuccess && count > 0);
    g_default_device = g_cuda_available ? DEVICE_CUDA : DEVICE_CPU;
#else
    g_cuda_available = 0;
    g_default_device = DEVICE_CPU;
#endif
}

int backend_cuda_available(void){
    return g_cuda_available;
}

Device backend_default_device(void){
    return g_default_device;
}

void backend_set_default(Device device){
    if(device == DEVICE_CUDA && !g_cuda_available){
        g_default_device = DEVICE_CPU;
        printf("Warning: CUDA is not available. Default device set to CPU.\n");
        return;
    }
    g_default_device = device;
}

float* cuda_malloc(int n){
#ifdef PINN_USE_CUDA
    float *ptr = NULL;
    cudaError_t err = cudaMalloc((void**)&ptr, n * sizeof(float));
    if(err != cudaSuccess){
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
#else
    (void)n;
    return NULL;
#endif
}

void cuda_free(float *ptr){
#ifdef PINN_USE_CUDA
    if(ptr){
        cudaFree(ptr);
    }
#else
    (void)ptr;
#endif
}

void cuda_memcpy_to_device(const float *src, float *dst, int n){
#ifdef PINN_USE_CUDA
    cudaError_t err = cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyHostToDevice);
    if(err != cudaSuccess){
        printf("CUDA memcpy to device failed: %s\n", cudaGetErrorString(err));
    }
#else
    (void)src;
    (void)dst;
    (void)n;
#endif
}

void cuda_memcpy_to_host(const float *src, float *dst, int n){
#ifdef PINN_USE_CUDA
    cudaError_t err = cudaMemcpy(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost);
    if(err != cudaSuccess){
        printf("CUDA memcpy to host failed: %s\n", cudaGetErrorString(err));
    }
#else
    (void)src;
    (void)dst;
    (void)n;
#endif
}
