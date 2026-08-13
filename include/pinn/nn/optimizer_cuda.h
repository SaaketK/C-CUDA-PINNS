#ifndef OPTIMIZER_CUDA_H
#define OPTIMIZER_CUDA_H

#ifdef __cplusplus
extern "C" {
#endif

int cuda_sgd_step(float *data, const float *grad, float lr, int size);

int cuda_adam_step(float *data, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float eps, float beta1_correction, float beta2_correction, int size);

#ifdef __cplusplus
}
#endif

#endif