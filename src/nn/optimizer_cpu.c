#include "pinn/nn/optimizer_cpu.h"
#include <math.h>

int cpu_sgd_step(float *data, const float *grad, float lr, int size){
    if(!data || !grad || size < 0) return 1;
    if(size == 0 || lr == 0.0f) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        data[i] -= lr * grad[i];
    }
    return 0;
}

int cpu_adam_step(float *data, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float eps, float beta1_correction, float beta2_correction, int size){
    if(!data || !grad || !m || !v || size < 0 || beta1_correction <= 0.0f || beta2_correction <= 0.0f) return 1;
    if(size == 0) return 0;

    #pragma omp parallel for
    for(int i = 0; i < size; i++){
        float g = grad[i];
        m[i] = beta1 * m[i] + (1.0f - beta1) * g;
        v[i] = beta2 * v[i] + (1.0f - beta2) * g * g;
        data[i] -= lr * (m[i] / beta1_correction) / (sqrtf(v[i] / beta2_correction) + eps);
    }
    return 0;
}
