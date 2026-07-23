/*
 * core/backend.h
 *
 * Owns backend/device declarations. This will eventually expose CPU/CUDA
 * dispatch helpers, cuda_to_cpu, cuda_to_cuda, device synchronization, and
 * backend capability checks.
*/

#ifndef BACKEND_H
#define BACKEND_H

#include "pinn/core/tensor.h"

int backend_init(void);
int backend_initialized(void);

int backend_cuda_compiled(void);
int backend_cuda_available(void);
int backend_device_count(void);

Device backend_default_device(void);
void backend_set_default(Device device);

void backend_sync(Device device);


// Memory Management
float* cuda_malloc(int n);
void cuda_free(float *ptr);
void cuda_memcpy_to_device(const float *src, float *dst, int n);
void cuda_memcpy_to_host(const float *src, float *dst, int n);

#endif
