/*
 * pinn/sampler.c
 *
 * Implements sampling utilities for PINN collocation and constraint points.
 * These functions generate coordinate tensors over simple domains; model
 * evaluation, residual computation, and boundary/initial targets are handled
 * by the trainer and residual layers.
*/

#include <stdio.h>
#include <stdlib.h>
#include "pinn/pinn/sampler.h"
#include "pinn/core/tensor.h"

Tensor* sample_uniform_box(BoxDomain *domain, int n_points){
    int shape[2] = {n_points, domain->dim};
    Tensor *data = tensor_create(shape, 2, 0);
    for(int i = 0; i < n_points; i++){
        for(int j = 0; j < shape[1]; j++){
            float r = (float) rand() / RAND_MAX;
            data->data[i * shape[1] + j] = domain->lower[j] + r * (domain->upper[j] - domain->lower[j]);
        }
    }
    return data;
}

Tensor* sample_fixed_dim_box(BoxDomain *domain, int n_points, int fixed_dim, float fixed_value){
    int shape[2] = {n_points, domain->dim};
    Tensor *data = tensor_create(shape, 2, 0);
    for(int i = 0; i < n_points; i++){
        for(int j = 0; j < shape[1]; j++){
            if(j == fixed_dim){
                data->data[i * shape[1] + j] = fixed_value;
            }
            else {
                float r = (float) rand() / RAND_MAX;
                data->data[i * shape[1] + j] = domain->lower[j] + r * (domain->upper[j] - domain->lower[j]);
            }
        }
    }
    return data;
}

Tensor* sample_LHS_box(BoxDomain *domain, int n){
    int shape[] = {n, domain->dim};
    Tensor *samples = tensor_create(shape, 2, 0);
    for(int i = 0; i < domain->dim; i++){
        float lower = domain->lower[i];
        float upper = domain->upper[i];
        float *values = malloc(n * sizeof(float));
        for(int j = 0; j < n; j++){
            float u = (float) rand() / (float) RAND_MAX;
            float sample = ((float) j + u) / (float) n;
            values[j] = lower + sample * (upper - lower);
        }
        for(int j = 0; j < n - 1; j++){
            int k = rand() % (j + 1);
            float temp = values[j];
            values[j] = values[k];
            values[k] = temp;
        }
        for(int row = 0; row < n; row++){
            samples->data[row * domain->dim + i] = values[row];
        }
        free(values);
    }
    return samples;
}