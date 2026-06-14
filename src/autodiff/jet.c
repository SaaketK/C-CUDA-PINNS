/*
 * autodiff/jet.c
 *
 * Implements forward-mode automatic differentiation for PDE input derivatives.
 * JetTensor values carry first- and second-order derivatives with respect to
 * model inputs such as t, x, S, or other equation coordinates. These routines
 * are used to compute PDE terms like u_t, u_xx, V_S, and V_SS while reverse-mode
 * autograd remains responsible for training neural-network weights.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pinn/autodiff/jet.h"
#include "pinn/core/ops.h"
#include "pinn/nn/mlp.h"

JetTensor* jet_create(Tensor *value, int input_dim){
    JetTensor *jet = malloc(sizeof(JetTensor));
    jet->value = value;
    jet->d1 = calloc(input_dim * value->size, sizeof(float));
    jet->d2 = calloc(input_dim * input_dim * value->size, sizeof(float));
    jet->input_dim = input_dim;
    jet->size = value->size;
    return jet;
}

JetTensor* jet_create_input(Tensor *value, int input_dim){
    JetTensor *jet = jet_create(value, input_dim);
    int batch = value->shape[0];
    for(int i = 0; i < batch; i++){
        for(int j = 0; j < input_dim; j++){
            int val_index = i * input_dim + j;
            jet_set_d1(jet, val_index, j, 1.0f);
        }
    }
    return jet;
}

void jet_free(JetTensor *jet){
    if(!jet) return;
    tensor_free(jet->value);
    free(jet->d1);
    free(jet->d2);
    free(jet);
}

// Getters & Setters

float jet_get_d1(JetTensor *jet, int value_index, int input_index){
    return jet->d1[value_index * jet->input_dim + input_index];
}

float jet_get_d2(JetTensor *jet, int value_index, int i, int j){
    return jet->d2[value_index * jet->input_dim * jet->input_dim + i * jet->input_dim + j];
}

void jet_set_d1(JetTensor *jet, int value_index, int input_index, float value){
    jet->d1[value_index * jet->input_dim + input_index] = value;
}

void jet_set_d2(JetTensor *jet, int value_index, int i, int j, float value){
    jet->d2[value_index * jet->input_dim * jet->input_dim + i * jet->input_dim + j] = value;
}

// Ops

JetTensor* jet_add(JetTensor *a, JetTensor *b){
    JetTensor *jet = jet_create(tensor_add(a->value, b->value), a->input_dim);
    for(int i = 0; i < jet->input_dim * jet->size; i++){
        jet->d1[i] = a->d1[i] + b->d1[i];
    }
    for(int i = 0; i < jet->input_dim * jet->input_dim * jet->size; i++){
        jet->d2[i] = a->d2[i] + b->d2[i];
    }
    return jet;
}

JetTensor* jet_matmult(JetTensor *a, Tensor *W){
    JetTensor *jet = jet_create(tensor_matmult(a->value, W), a->input_dim);
    int batch = a->value->shape[0];
    int in_features = a->value->shape[1];
    int out_features = W->shape[1];
    for(int r = 0; r < batch; r++){
        for(int c = 0; c < out_features; c++){
            int out_index = r * out_features + c;
            for(int p = 0; p < a->input_dim; p++){
                float sum = 0.0f;
                for(int k = 0; k < in_features; k++){
                    int in_index = r * in_features + k;
                    sum += jet_get_d1(a, in_index, p) * W->data[k * out_features + c];
                }
                jet_set_d1(jet, out_index, p, sum);
            }
            for(int p = 0; p < a->input_dim; p++){
                for(int q = 0; q < a->input_dim; q++){
                    float sum = 0.0f;
                    for(int k = 0; k < in_features; k++){
                        int in_index = r * in_features + k;
                        sum += jet_get_d2(a, in_index, p, q) * W->data[k * out_features + c];
                    }
                    jet_set_d2(jet, out_index, p, q, sum);
                }
            }
        }
    }
    return jet;
}

JetTensor* jet_square(JetTensor *a){
    JetTensor *jet = jet_create(tensor_square(a->value), a->input_dim);
    for(int v = 0; v < a->size; v++){
        float x = a->value->data[v];
        for(int p = 0; p < a->input_dim; p++){
            float x_p = jet_get_d1(a, v, p);
            jet_set_d1(jet, v, p, 2.0f * x * x_p);
        }
        for(int p = 0; p < a->input_dim; p++){
            for(int q = 0; q < a->input_dim; q++){
                float x_p = jet_get_d1(a, v, p);
                float x_q = jet_get_d1(a, v, q);
                float x_pq = jet_get_d2(a, v, p, q);

                float y_pq = 2.0f * x_p * x_q + 2.0f * x * x_pq;
                jet_set_d2(jet, v, p, q, y_pq);
            }
        }
    }
    return jet;
}

JetTensor* jet_tanh(JetTensor *a){
    JetTensor *jet = jet_create(tensor_tanh(a->value), a->input_dim);
    for(int v = 0; v < a->size; v++){
        float yval = jet->value->data[v];
        float f1 = 1.0f - yval * yval;
        float f2 = -2.0f * yval * f1;
        for(int p = 0; p < a->input_dim; p++){
            float x_p = jet_get_d1(a, v, p);
            jet_set_d1(jet, v, p, f1 * x_p);
        }
        for(int p = 0; p < a->input_dim; p++){
            for(int q = 0; q < a->input_dim; q++){
                float x_p = jet_get_d1(a, v, p);
                float x_q = jet_get_d1(a, v, q);
                float x_pq = jet_get_d2(a, v, p, q);

                float y_pq = f2 * x_p * x_q + f1 * x_pq;
                jet_set_d2(jet, v, p, q, y_pq);
            }
        }
    }
    return jet;
}

JetTensor* jet_bias_add(JetTensor *a, Tensor *b){
    JetTensor *jet = jet_create(tensor_bias_add(a->value, b), a->input_dim);

    int d1_count = a->size * a->input_dim;
    int d2_count = a->size * a->input_dim * a->input_dim;

    for(int i = 0; i < d1_count; i++){
        jet->d1[i] = a->d1[i];
    }

    for(int i = 0; i < d2_count; i++){
        jet->d2[i] = a->d2[i];
    }

    return jet;
}

JetTensor* jet_mlp_forward(MLP *mlp, JetTensor *x){
    JetTensor *out = x;
    for(int i = 0; i < mlp->n_layers; i++){
        out = jet_matmult(out, mlp->layers[i]->W);
        out = jet_bias_add(out, mlp->layers[i]->b);
        if(i != mlp->n_layers - 1){
            out = jet_tanh(out);
        }
    }
    return out;
}