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
static void scalar_mult_backward(Node *node);
static void scalar_add_backward(Node *node);
static void identity_backward(Node *node);
static void scale_deriv_backward(Node *node);
static void chain_d2_backward(Node *node);
static void deriv_matmult_backward(Node *node);
static void select_d1_backward(Node *node);
static void select_d2_backward(Node *node);

// add to tape

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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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
        node->ctx = NULL;
        node->free_ctx = NULL;
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

Tensor* tensor_scalar_mult(Tensor *a, float scalar){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    tape_record_tensor(output);
    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] * scalar;
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = scalar_mult_backward;
        ScalarCtx *ctx = malloc(sizeof(ScalarCtx));
        ctx->scalar = scalar;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_scalar_add(Tensor *a, float scalar){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    tape_record_tensor(output);
    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i] + scalar;
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = scalar_add_backward;
        node->ctx = NULL;
        node->free_ctx = NULL;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_identity(Tensor *a){
    Tensor *output = tensor_create(a->shape, a->ndim, a->req_grad);
    tape_record_tensor(output);
    for(int i = 0; i < a->size; i++){
        output->data[i] = a->data[i];
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = identity_backward;
        node->ctx = NULL;
        node->free_ctx = NULL;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_scale_deriv(Tensor *deriv, Tensor *factor, int input_dim){
    Tensor *output = tensor_create(deriv->shape, deriv->ndim, deriv->req_grad || factor->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);
    for(int i = 0; i < factor->size; i++){
        for(int j = 0; j < input_dim; j++){
            int idx = i * input_dim + j;
            output->data[idx] = deriv->data[idx] * factor->data[i];
        }
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = deriv;
        node->inputs[1] = factor;
        node->n_inputs = 2;
        node->output = output;
        node->backward = scale_deriv_backward;
        ScaleDerivCtx *ctx = malloc(sizeof(ScaleDerivCtx));
        ctx->input_dim = input_dim;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_chain_d2(Tensor *d1, Tensor *d2, Tensor *f_prime, Tensor *f_double, int input_dim){
    Tensor *output = tensor_create(d2->shape, d2->ndim, d1->req_grad || d2->req_grad || f_prime->req_grad || f_double->req_grad);
    tape_record_tensor(output);
    for(int i = 0; i < f_prime->size; i++){
        for(int p = 0; p < input_dim; p++){
            for(int q = 0; q < input_dim; q++){
                int idx2 = i * input_dim * input_dim + p * input_dim + q;
                int idxp = i * input_dim + p;
                int idxq = i * input_dim + q;
                output->data[idx2] = f_double->data[i] * d1->data[idxp] * d1->data[idxq] + f_prime->data[i] * d2->data[idx2];
            }
        }
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(4 * sizeof(Tensor*));
        node->inputs[0] = d1;
        node->inputs[1] = d2;
        node->inputs[2] = f_prime;
        node->inputs[3] = f_double;
        node->n_inputs = 4;
        node->output = output;
        node->backward = chain_d2_backward;
        ChainD2Ctx *ctx = malloc(sizeof(ChainD2Ctx));
        ctx->input_dim = input_dim;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

static Tensor* tensor_deriv_matmult_order(Tensor *deriv, Tensor *W, int batch, int in_features, int out_features, int input_dim, int order){
    int channels = 1;
    for(int i = 0; i < order; i++){
        channels *= input_dim;
    }
    int output_shape[2] = {batch * out_features, channels};
    Tensor *output = tensor_create(output_shape, 2, deriv->req_grad || W->req_grad);
    if(!output) return NULL;
    tape_record_tensor(output);

    for(int r = 0; r < batch; r++){
        for(int c = 0; c < out_features; c++){
            int out_value_index = r * out_features + c;
            for(int ch = 0; ch < channels; ch++){
                float sum = 0.0f;
                for(int k = 0; k < in_features; k++){
                    int in_value_index = r * in_features + k;
                    sum += deriv->data[in_value_index * channels + ch] * W->data[k * out_features + c];
                }
                output->data[out_value_index * channels + ch] = sum;
            }
        }
    }

    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(2 * sizeof(Tensor*));
        node->inputs[0] = deriv;
        node->inputs[1] = W;
        node->n_inputs = 2;
        node->output = output;
        node->backward = deriv_matmult_backward;
        DerivMatmultCtx *ctx = malloc(sizeof(DerivMatmultCtx));
        ctx->batch = batch;
        ctx->in_features = in_features;
        ctx->out_features = out_features;
        ctx->input_dim = input_dim;
        ctx->order = order;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_deriv_matmult(Tensor *deriv, Tensor *W, int batch, int in_features, int out_features, int input_dim){
    return tensor_deriv_matmult_order(deriv, W, batch, in_features, out_features, input_dim, 1);
}

Tensor* tensor_deriv2_matmult(Tensor *deriv, Tensor *W, int batch, int in_features, int out_features, int input_dim){
    return tensor_deriv_matmult_order(deriv, W, batch, in_features, out_features, input_dim, 2);
}

Tensor* tensor_select_d1(Tensor *d1, int input_dim, int component){
    int out_dim = d1->shape[1] / input_dim;
    int shape[] = {d1->shape[0], out_dim};
    Tensor *output = tensor_create(shape, 2, d1->req_grad);
    tape_record_tensor(output);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < out_dim; j++){
            output->data[i * out_dim + j] = d1->data[(i * out_dim + j) * input_dim + component];
        }
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(1 * sizeof(Tensor*));
        node->inputs[0] = d1;
        node->n_inputs = 1;
        node->output = output;
        node->backward = select_d1_backward;
        SelectD1Ctx *ctx = malloc(sizeof(SelectD1Ctx));
        ctx->input_dim = input_dim;
        ctx->component = component;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_select_d2(Tensor *d2, int input_dim, int p, int q){
    int out_dim = d2->shape[1]  / (input_dim * input_dim);
    int shape[] = {d2->shape[0], out_dim};
    Tensor *output = tensor_create(shape, 2, d2->req_grad);
    tape_record_tensor(output);
    for(int i = 0; i < shape[0]; i++){
        for(int j = 0; j < out_dim; j++){
            output->data[i * out_dim + j] = d2->data[(i * out_dim + j) * input_dim * input_dim + p * input_dim +q];
        }
    }
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = d2;
        node->n_inputs = 1;
        node->output = output;
        node->backward = select_d2_backward;
        SelectD2Ctx * ctx = malloc(sizeof(SelectD2Ctx));
        ctx->input_dim = input_dim;
        ctx->p = p;
        ctx->q = q;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
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

static void scalar_mult_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    ScalarCtx *ctx = (ScalarCtx*)node->ctx;
    if(a->req_grad){
        for(int i = 0; i < a->size; i++){
            a->grad[i] += output->grad[i] * ctx->scalar;
        }
    }
}

static void scalar_add_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    if(a->req_grad){
        for(int i = 0;i < a->size; i++){
            a->grad[i] += output->grad[i];
        }
    }
}

static void identity_backward(Node *node){
    // Same math
    scalar_add_backward(node);
}

static void scale_deriv_backward(Node *node){
    Tensor *deriv = node->inputs[0];
    Tensor *factor = node->inputs[1];
    Tensor *output = node->output;
    ScaleDerivCtx *ctx = (ScaleDerivCtx*)node->ctx;
    int input_dim = ctx->input_dim;
    for(int i = 0; i < factor->size; i++){
        for(int j = 0;  j < input_dim; j++){
            if(deriv->req_grad){
                deriv->grad[i * input_dim + j] += output->grad[i * input_dim + j] * factor->data[i];
            }
            if(factor->req_grad){
                factor->grad[i] += output->grad[i * input_dim + j] * deriv->data[i * input_dim + j];
            }
        }
    }
}

static void chain_d2_backward(Node *node){
    Tensor *d1 = node->inputs[0];
    Tensor *d2 = node->inputs[1];
    Tensor *f_prime = node->inputs[2];
    Tensor *f_double = node->inputs[3];
    Tensor *output = node->output;
    ChainD2Ctx *ctx = (ChainD2Ctx*)node->ctx;
    int input_dim = ctx->input_dim;
    for(int i = 0; i < f_prime->size; i++){
        for(int p = 0; p < input_dim; p++){
            for(int q = 0; q < input_dim; q++){
                int idx2 = i * input_dim * input_dim + p * input_dim + q;
                int idxp = i * input_dim + p;
                int idxq = i * input_dim + q;
                if(d1->req_grad){
                    d1->grad[idxp] += output->grad[idx2] * f_double->data[i] * d1->data[idxq];
                    d1->grad[idxq] += output->grad[idx2] * f_double->data[i] * d1->data[idxp];
                }
                if(d2->req_grad){
                    d2->grad[idx2] += output->grad[idx2] * f_prime->data[i];
                }
                if(f_prime->req_grad){
                    f_prime->grad[i] += output->grad[idx2] * d2->data[idx2];
                }
                if(f_double->req_grad){
                    f_double->grad[i] += output->grad[idx2] * d1->data[idxp] * d1->data[idxq];
                }
            }
        }
    }
}

static void deriv_matmult_backward(Node *node){
    Tensor *deriv = node->inputs[0];
    Tensor *W = node->inputs[1];
    Tensor *output = node->output;
    DerivMatmultCtx *ctx = (DerivMatmultCtx*)node->ctx;
    int channels = 1;
    for(int i = 0; i < ctx->order; i++){
        channels *= ctx->input_dim;
    }

    for(int r = 0; r < ctx->batch; r++){
        for(int c = 0; c < ctx->out_features; c++){
            int out_value_index = r * ctx->out_features + c;
            for(int ch = 0; ch < channels; ch++){
                float grad = output->grad[out_value_index * channels + ch];
                for(int k = 0; k < ctx->in_features; k++){
                    int in_value_index = r * ctx->in_features + k;
                    int deriv_idx = in_value_index * channels + ch;
                    int w_idx = k * ctx->out_features + c;
                    if(deriv->req_grad){
                        deriv->grad[deriv_idx] += grad * W->data[w_idx];
                    }
                    if(W->req_grad){
                        W->grad[w_idx] += grad * deriv->data[deriv_idx];
                    }
                }
            }
        }
    }
}

static void select_d1_backward(Node *node){
    Tensor *d1 = node->inputs[0];
    Tensor *output = node->output;
    SelectD1Ctx *ctx = (SelectD1Ctx*)node->ctx;
    int input_dim = ctx->input_dim;
    int component = ctx->component;
    int output_dim = d1->shape[1] / input_dim;
    for(int i = 0; i < d1->shape[0]; i++){
        for(int j = 0; j < output_dim; j++){
            if(d1->req_grad){
                d1->grad[(i * output_dim + j) * input_dim + component] += output->grad[i * output_dim + j];
            }
        }
    }
}

static void select_d2_backward(Node *node){
    Tensor *d2 = node->inputs[0];
    Tensor *output = node->output;
    SelectD2Ctx *ctx = (SelectD2Ctx*)node->ctx;
    int input_dim = ctx->input_dim;
    int p = ctx->p;
    int q = ctx->q;
    int out_dim = d2->shape[1] / (input_dim * input_dim);
    for(int i = 0; i < d2->shape[0]; i++){
        for(int j = 0; j < out_dim; j++){
            d2->grad[(i * out_dim + j) * input_dim * input_dim + p * input_dim +q] += output->grad[i * out_dim + j];
        }
    }
}