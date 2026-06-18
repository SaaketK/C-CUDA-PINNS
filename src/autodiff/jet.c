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
    int d1_shape[2] = {value->size, input_dim};
    int d2_shape[3] = {value->size, input_dim, input_dim};
    jet->d1 = tensor_create(d1_shape, 2, value->req_grad);
    jet->d2 = tensor_create(d2_shape, 3, value->req_grad);
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
    tensor_free(jet->d1);
    tensor_free(jet->d2);
    free(jet);
}

void jet_free_shallow(JetTensor *jet){
    if(!jet) return;
    free(jet);
}

// Getters & Setters

float jet_get_d1(JetTensor *jet, int value_index, int input_index){
    return jet->d1->data[value_index * jet->input_dim + input_index];
}

float jet_get_d2(JetTensor *jet, int value_index, int i, int j){
    return jet->d2->data[value_index * jet->input_dim * jet->input_dim + i * jet->input_dim + j];
}

void jet_set_d1(JetTensor *jet, int value_index, int input_index, float value){
    jet->d1->data[value_index * jet->input_dim + input_index] = value;
}

void jet_set_d2(JetTensor *jet, int value_index, int i, int j, float value){
    jet->d2->data[value_index * jet->input_dim * jet->input_dim + i * jet->input_dim + j] = value;
}

// Ops

JetTensor* jet_add(JetTensor *a, JetTensor *b){
    Tensor *value = tensor_add(a->value, b->value);
    Tensor *d1 = tensor_add(a->d1, b->d1);
    Tensor *d2 = tensor_add(a->d2, b->d2);
    JetTensor *jet = jet_from_parts(value, d1, d2, a->input_dim);
    return jet;
}

JetTensor* jet_matmult(JetTensor *a, Tensor *W){
    int batch = a->value->shape[0];
    int in_features = a->value->shape[1];
    int out_features = W->shape[1];
    Tensor *value = tensor_matmult(a->value, W);
    Tensor *d1 = tensor_deriv_matmult(a->d1, W, batch, in_features, out_features, a->input_dim);
    Tensor *d2 = tensor_deriv2_matmult(a->d2, W, batch, in_features, out_features, a->input_dim);
    JetTensor *jet = jet_from_parts(value, d1, d2, a->input_dim);
    return jet;
}

JetTensor* jet_square(JetTensor *a){
    Tensor *val = tensor_square(a->value);
    Tensor *f_prime = tensor_scalar_mult(a->value, 2.0f);
    Tensor *d1 = tensor_scale_deriv(a->d1, f_prime, a->input_dim);
    Tensor *f_double = tensor_scalar_add(tensor_scalar_mult(a->value, 0.0f), 2.0f);
    Tensor *d2 = tensor_chain_d2(a->d1, a->d2, f_prime, f_double, a->input_dim);
    JetTensor *jet = jet_from_parts(val, d1, d2, a->input_dim);
    return jet;
}

JetTensor* jet_tanh(JetTensor *a){
    Tensor *val = tensor_tanh(a->value);
    Tensor *val_sq = tensor_square(val);
    Tensor *factor = tensor_scalar_add(tensor_scalar_mult(val_sq, -1.0f), 1.0f);
    Tensor *d1 = tensor_scale_deriv(a->d1, factor, a->input_dim);
    Tensor *d2 = tensor_chain_d2(a->d1, a->d2, factor, tensor_mult(tensor_scalar_mult(val, -2.0f), factor), a->input_dim);
    JetTensor *jet = jet_from_parts(val, d1, d2, a->input_dim);
    return jet;
}

JetTensor* jet_bias_add(JetTensor *a, Tensor *b){
    Tensor *value = tensor_bias_add(a->value, b);
    Tensor *d1 = tensor_identity(a->d1);
    Tensor *d2 = tensor_identity(a->d2);
    JetTensor *jet = jet_from_parts(value, d1, d2, a->input_dim);
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

// Jet Free with Jet Tape

static JetTape *curr_jet_tape = NULL;

JetTape *jet_tape_create(void){
    JetTape *tape = malloc(sizeof(JetTape));
    tape->items = NULL;
    tape->size = 0;
    tape->capacity = 0;
    return tape;
}

int jet_tape_add(JetTape *tape, JetTensor *item){
    if(!tape || !item) return 0;

    if(tape->size == tape->capacity){
        int new_capacity = tape->capacity == 0 ? 16 : tape->capacity * 2;
        JetTensor **new_items = realloc(tape->items, new_capacity * sizeof(JetTensor*));
        if(!new_items) return 0;
        tape->items = new_items;
        tape->capacity = new_capacity;
    }
    tape->items[tape->size] = item;
    tape->size++;
    return 1;
}

JetTape* get_curr_jet_tape(void){
    return curr_jet_tape;
}

void set_curr_jet_tape(JetTape *tape){
    curr_jet_tape = tape;
}

void jet_tape_free(JetTape *tape){
    if(!tape) return;
    if (curr_jet_tape == tape) curr_jet_tape = NULL;
    for(int i = 0; i < tape->size; i++){
        jet_free(tape->items[i]);
    }
    free(tape->items);
    free(tape);
}

void jet_tape_free_shallow(JetTape *tape){
    if(!tape) return;
    if(curr_jet_tape == tape) curr_jet_tape = NULL;
    for(int i = 0; i < tape->size; i++){
        jet_free_shallow(tape->items[i]);
    }
    free(tape->items);
    free(tape);
}

// Graph construction helper

JetTensor* jet_from_parts(Tensor *value, Tensor *d1, Tensor *d2, int input_dim){
    JetTensor *jet = malloc(sizeof(JetTensor));
    jet->value = value;
    jet->d1 = d1;
    jet->d2 = d2;
    jet->input_dim = input_dim;
    jet->size = value->size;
    JetTape *tape = get_curr_jet_tape();
    if(tape){
        jet_tape_add(tape, jet);
    }
    return jet;
}
