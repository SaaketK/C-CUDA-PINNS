#ifndef OPTIMIZER_CPU_H
#define OPTIMIZER_CPU_H

int cpu_sgd_step(float *data, const float *grad, float lr, int size);

int cpu_adam_step(float *data, const float *grad, float *m, float *v, float lr, float beta1, float beta2, float eps, float beta1_correction, float beta2_correction, int size);

#endif