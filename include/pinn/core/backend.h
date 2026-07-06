/*
 * core/backend.h
 *
 * Owns backend/device declarations. This will eventually expose CPU/CUDA
 * dispatch helpers, cuda_to_cpu, cuda_to_cuda, device synchronization, and
 * backend capability checks.
*/

#include "pinn/core/tensor.h"

#ifndef BACKEND_H
#define BACKEND_H

void backend_init();
int backend_cuda_available();
Device backend_default_device();
void backend_set_default(Device device);


// Memory Management
float* cuda_malloc(int n);
void cuda_free(float *ptr);
void cuda_memcpy_to_device(const float *src, float *dst, int n);
void cuda_memcpy_to_host(const float *src, float *dst, int n);

#endif