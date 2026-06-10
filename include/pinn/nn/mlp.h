/*
 * nn/mlp.h
 *
 * neural-network model declarations: Linear layers, MLP containers,
 * parameter collection, forward passes, initialization, and free functions.
*/

#ifndef MLP_H
#define MLP_H

#include "pinn/core/tensor.h"

// Individual Layer

typedef struct {
    Tensor *W;
    Tensor *b;
    int input_dim;
    int output_dim;
} Linear;

Linear* linear_create(int input_dim, int output_dim);
Tensor* linear_forward(Linear *layer, Tensor *x);
void linear_free(Linear *layer);

// Multi Layer

typedef struct {
    Linear **layers;
    int n_layers;
} MLP;

MLP* mlp_create(int *layer_sizes, int n_sizes);
Tensor* mlp_forward(MLP *mlp, Tensor *x);
Tensor** mlp_parameters(MLP *mlp, int *n_params);
void mlp_free(MLP *mlp);

#endif