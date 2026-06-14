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
