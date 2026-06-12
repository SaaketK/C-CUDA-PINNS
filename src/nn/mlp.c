/*
 * nn/mlp.c
 *
 * Implements neural network model pieces: Linear layer allocation,
 * initialization, forward passes built from core tensor ops, parameter
 * collection, and model/layer cleanup.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pinn/nn/mlp.h"
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"

Linear* linear_create(int input_dim, int output_dim){
    Linear *layer = malloc(sizeof(Linear));
    layer->input_dim = input_dim;
    layer->output_dim = output_dim;
    int W_shape[2] = {input_dim, output_dim};
    int b_shape[1] = {output_dim};
    layer->W = tensor_create(W_shape, 2, 1);
    layer->b = tensor_create(b_shape, 1, 1);
    float limit = sqrtf(6.0f / (input_dim + output_dim));
    for(int i = 0; i < layer->W->size; i++){
        // Use Xavier Initialization later
        float r = (float)rand() / (float)RAND_MAX; // [0, 1]
        layer->W->data[i] = limit * (2.0f * r - 1.0f); // [-limit, limit]

    }
    return layer;
}

Tensor* linear_forward(Linear *layer, Tensor *x){
    Tensor *z = tensor_matmult(x, layer->W);
    Tensor *y = tensor_bias_add(z, layer->b);
    return y;
}

void linear_free(Linear *layer){
    tensor_free(layer->W);
    tensor_free(layer->b);
    free(layer);
}

MLP* mlp_create(int *layer_sizes, int n_sizes){
    MLP* mlp = malloc(sizeof(MLP));
    mlp->n_layers = n_sizes - 1;
    mlp->layers = malloc(mlp->n_layers * sizeof(Linear*));
    for(int i = 0; i < mlp->n_layers; i++){
        mlp->layers[i] = linear_create(layer_sizes[i], layer_sizes[i+1]);
    }
    return mlp;
}

Tensor* mlp_forward(MLP *mlp, Tensor *x){
    Tensor *out = x;
    for(int i = 0; i < mlp->n_layers; i++){
        out = linear_forward(mlp->layers[i], out);
        if(i != mlp->n_layers - 1){
            out = tensor_tanh(out);
        }
    }
    return out;
}

void mlp_free(MLP *mlp){
    for(int i = 0; i < mlp->n_layers; i++){
        linear_free(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

Tensor** mlp_parameters(MLP *mlp, int *n_params){
    *n_params = 2 * mlp->n_layers;
    Tensor **params = malloc(*n_params * sizeof(Tensor*));
    for(int i = 0; i < mlp->n_layers; i++){
        params[2*i] = mlp->layers[i]->W;
        params[2*i + 1] = mlp->layers[i]->b;
    }
    return params;
}