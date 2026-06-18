/*
 * autodiff/jet.h
 *
 * Derivatives to compute the Physics Residual
 * Owns the JetTensor API for PINN input derivatives: model values, first
 * derivatives, and full Hessians with respect to low-dimensional PDE inputs.
 * This layer provides equation-agnostic derivative accessors for residuals.
*/

#ifndef JET_H
#define JET_H

#include "pinn/core/tensor.h"
#include "pinn/nn/mlp.h"

typedef struct {
    Tensor *value;
    Tensor *d1;
    Tensor *d2;
    int input_dim;
    int size;
} JetTensor;

JetTensor* jet_create(Tensor *value, int input_dim);
JetTensor* jet_create_input(Tensor *value, int input_dim);
void jet_free(JetTensor *jet);
void jet_free_shallow(JetTensor *jet);

// Getters & Setters

float jet_get_d1(JetTensor *jet, int value_index, int input_index);
float jet_get_d2(JetTensor *jet, int value_index, int i, int j);
void jet_set_d1(JetTensor *jet, int value_index, int input_index, float value);
void jet_set_d2(JetTensor *jet, int value_index, int i, int j, float value);

// Ops

JetTensor* jet_add(JetTensor *a, JetTensor *b);
JetTensor* jet_matmult(JetTensor *a, Tensor *W);
JetTensor* jet_square(JetTensor *a);
JetTensor* jet_tanh(JetTensor *a);
JetTensor* jet_bias_add(JetTensor *a, Tensor *b);
JetTensor* jet_mlp_forward(MLP *mlp, JetTensor *x);

// Graph Builder for backpropagation 

JetTensor* jet_from_parts(Tensor *value, Tensor *d1, Tensor *d2, int input_dim);

// Jet Free using Tape

typedef struct {
    JetTensor **items;
    int size;
    int capacity;
} JetTape;

JetTape *jet_tape_create(void);
int jet_tape_add(JetTape *tape, JetTensor *item);
void set_curr_jet_tape(JetTape *tape);
JetTape* get_curr_jet_tape(void);
void jet_tape_free(JetTape *tape);
void jet_tape_free_shallow(JetTape *tape);

#endif
