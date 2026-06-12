/*
 * core/ops.c
 *
 * Implements tensor operation forward passes and their backward functions.
 * Each differentiable op computes output data, saves needed context, creates
 * an autograd node, and accumulates gradients into input tensors during backward.
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/core/autograd.h"

static void mult_backward(Node *node);
static void add_backward(Node *node);
static void sub_backward(Node *node);
static void square_backward(Node *node);
static void mean_backward(Node *node);
static void tanh_backward(Node *node);
static void matmult_backward(Node *node);
static void bias_add_backward(Node *node);

static void tape_record_tensor(Tensor *tensor){
    Tape *tape = get_curr_tape();
    if(tape){
        tape_add_tensor(tape, tensor);
    }
}

static void tape_record_node(Node *node){
    Tape *tape = get_curr_tape();
    if(tape){
        tape_add_node(tape, node);
    }
}

// Forward Pass

Tensor* tensor_add(Tensor *a, Tensor *b){
    Tensor* output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);

    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] + b->data[i];
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = a;
        node->inputs[1] = b;
        node->n_inputs = 2;
        node->output = output;
        node->backward = add_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_mult(Tensor *a, Tensor *b){
    Tensor* output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);

    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] * b->data[i];
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = a;
        node->inputs[1] = b;
        node->n_inputs = 2;
        node->output = output;
        node->backward = mult_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_mean(Tensor *a){
    int shape[] = {1};
    Tensor *output = tensor_create(shape, 1, a->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);
    float sum = 0.0f;
    for(int i = 0; i < a->size; i++){
        sum += a->data[i];
    }
    output->data[0] = sum/a->size;
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = mean_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_sub(Tensor *a, Tensor *b){
    Tensor* output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);

    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] - b->data[i];
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = a;
        node->inputs[1] = b;
        node->n_inputs = 2;
        node->output = output;
        node->backward = sub_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_matmult(Tensor *a, Tensor *b){
    if(a->ndim != 2 || b->ndim != 2) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    int output_shape[] = {a->shape[0], b->shape[1]};
    Tensor *output = tensor_create(output_shape, 2, a->req_grad || b->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);

    int rows = a->shape[0];
    int inner = a->shape[1];
    int cols = b->shape[1];

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            float sum = 0.0f;
            for(int k = 0; k < inner; k++){
                sum += a->data[i * inner + k] * b->data[k * cols + j];
            }
            output->data[i * cols + j] = sum;
        }
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = a;
        node->inputs[1] = b;
        node->n_inputs = 2;
        node->output = output;
        node->backward = matmult_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_bias_add(Tensor *a, Tensor *b){
    if(a->ndim != 2 || b->ndim != 1) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad || b->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);

    int rows = a->shape[0];
    int cols = a->shape[1];

    for(int i = 0; i < rows; i++){
        for(int j = 0; j < cols; j++){
            output->data[i * cols + j] = a->data[i * cols + j] + b->data[j];
        }
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = a;
        node->inputs[1] = b;
        node->n_inputs = 2;
        node->output = output;
        node->backward = bias_add_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_square(Tensor *a){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);
    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] * a->data[i];
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = square_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}
Tensor* tensor_tanh(Tensor *a){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);
    for(int i = 0; i < a->size; i++){
        output->data[i] = tanh(a->data[i]);
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = tanh_backward;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_mse(Tensor *a, Tensor *b){
    Tensor *diff = tensor_sub(a, b);
    Tensor *sq = tensor_square(diff);
    Tensor *output = tensor_mean(sq);
    return output;
}

// Backward Pass

static void mult_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;
    
    for(int i = 0; i < output->size; i++){
        if(a->req_grad){
            a->grad[i] += output->grad[i] * b->data[i];
        }
        if(b->req_grad){
            b->grad[i] += output->grad[i] * a->data[i];
        }
    }
}

static void add_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;
    
    for(int i = 0; i < output->size; i++){
        if(a->req_grad){
            a->grad[i] += output->grad[i];
        }
        if(b->req_grad){
            b->grad[i] += output->grad[i];
        }
    }
}

static void sub_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;
    
    for(int i = 0; i < output->size; i++){
        if(a->req_grad){
            a->grad[i] += output->grad[i];
        }
        if(b->req_grad){
            b->grad[i] -= output->grad[i];
        }
    }
}

static void square_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    
    if(a->req_grad){
        for(int i = 0; i < a->size; i++){
            a->grad[i] += 2 * output->grad[i] * a->data[i];
        }
    }
}

static void mean_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    
    if(a->req_grad){
        for(int i = 0; i < a->size; i++){
            a->grad[i] += output->grad[0] / a->size;
        }
    }
}

static void tanh_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    
    if(a->req_grad){
        for(int i = 0; i < a->size; i++){
            a->grad[i] += output->grad[i] * (1.0f - output->data[i] * output->data[i]);
        }
    }
}

static void matmult_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *c = node->output;
    int rows = a->shape[0];
    int inner = a->shape[1];
    int cols = b->shape[1];

    if(a->req_grad){   
        for(int i = 0; i < rows; i++){
            for(int k = 0; k < inner; k++){
                for(int j = 0; j < cols; j++){
                    a->grad[i * inner + k] += c->grad[i * cols + j] * b->data[k * cols + j];
                }
            }
        }
    }
    if(b->req_grad){
        for(int k = 0; k < inner; k++){
            for(int j = 0; j < cols; j++){
                for(int i = 0; i < rows; i++){
                    b->grad[k * cols + j] += c->grad[i * cols + j] * a->data[i * inner + k];
                }
            }
        }
    }
}


static void bias_add_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;
    int rows = a->shape[0];
    int cols = a->shape[1];

    if(a->req_grad){
        for(int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                a->grad[i * cols + j] += output->grad[i * cols + j];
            }
        }
    }
    if(b->req_grad){
        for(int j = 0; j < cols; j++){
            float sum = 0.0f;
            for(int i = 0; i < rows; i++){
                sum += output->grad[i * cols + j];
            }
            b->grad[j] += sum;
        }
    }
}
