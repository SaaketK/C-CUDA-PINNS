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
#include "pinn/core/ops_cpu.h"
#include "pinn/core/autograd.h"
#include "pinn/core/backend.h"

#ifdef PINN_USE_CUDA
#include "pinn/core/ops_cuda.h"
#endif

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
static void select_col_backward(Node *node);
static void relu_backward(Node *node);
static void sigmoid_backward(Node *node);

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
    if(!a || !b || a->size != b->size || a->device != b->device) return NULL;

    Tensor* output = tensor_create_device(a->shape, a->ndim, a->req_grad || b->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

    #ifdef PINN_USE_CUDA
        if(a->device == DEVICE_CUDA && backend_cuda_available()){
            status = cuda_add(a->data, b->data, output->data, a->size);
        }
        else {
            status = cpu_add(a->data, b->data, output->data, a->size);
        }
    #else
        status = cpu_add(a->data, b->data, output->data, a->size);
    #endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a || !b || a->size != b->size || a->device != b->device) return NULL;

    Tensor* output = tensor_create_device(a->shape, a->ndim, a->req_grad || b->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

    #ifdef PINN_USE_CUDA
        if(a->device == DEVICE_CUDA && backend_cuda_available()){
            status = cuda_mult(a->data, b->data, output->data, a->size);
        }
        else {
            status = cpu_mul(a->data, b->data, output->data, a->size);
        }
    #else
        status = cpu_mul(a->data, b->data, output->data, a->size);
    #endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a) return NULL;

    int shape[] = {1};
    Tensor *output = tensor_create_device(shape, 1, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

    #ifdef PINN_USE_CUDA
        if(a->device == DEVICE_CUDA && backend_cuda_available()){
            status = cuda_mean(a->data, output->data, a->size);
        }
        else {
            status = cpu_mean(a->data, output->data, a->size);
        }
    #else
        status = cpu_mean(a->data, output->data, a->size);
    #endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a || !b || a->size != b->size || a->device != b->device) return NULL;

    Tensor* output = tensor_create_device(a->shape, a->ndim, a->req_grad || b->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

    #ifdef PINN_USE_CUDA
        if(a->device == DEVICE_CUDA && backend_cuda_available()){
            status = cuda_sub(a->data, b->data, output->data, a->size);
        }
        else {
            status = cpu_sub(a->data, b->data, output->data, a->size);
        }
    #else
        status = cpu_sub(a->data, b->data, output->data, a->size);
    #endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a || !b || a->device != b->device) return NULL;
    if(a->ndim != 2 || b->ndim != 2) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    int output_shape[] = {a->shape[0], b->shape[1]};
    Tensor *output = tensor_create_device(output_shape, 2, a->req_grad || b->req_grad, a->device);
    if(!output) return NULL;

    int rows = a->shape[0];
    int inner = a->shape[1];
    int cols = b->shape[1];
    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_matmult(a->data, b->data, output->data, rows, inner, cols);
    }
    else {
        status = cpu_matmult(a->data, b->data, output->data, rows, inner, cols);
    }
#else
    status = cpu_matmult(a->data, b->data, output->data, rows, inner, cols);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a || !b || a->device != b->device) return NULL;
    if(a->ndim != 2 || b->ndim != 1) return NULL;
    if(a->shape[1] != b->shape[0]) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad || b->req_grad, a->device);
    if(!output) return NULL;

    int rows = a->shape[0];
    int cols = a->shape[1];
    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_bias_add(a->data, b->data, output->data, rows, cols);
    }
    else {
        status = cpu_bias_add(a->data, b->data, output->data, rows, cols);
    }
#else
    status = cpu_bias_add(a->data, b->data, output->data, rows, cols);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_square(a->data, output->data, a->size);
    }
    else {
        status = cpu_square(a->data, output->data, a->size);
    }
#else
    status = cpu_square(a->data, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_tanh(a->data, output->data, a->size);
    }
    else {
        status = cpu_tanh(a->data, output->data, a->size);
    }
#else
    status = cpu_tanh(a->data, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_scalar_mult(a->data, scalar, output->data, a->size);
    }
    else {
        status = cpu_scalar_mult(a->data, scalar, output->data, a->size);
    }
#else
    status = cpu_scalar_mult(a->data, scalar, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_scalar_add(a->data, scalar, output->data, a->size);
    }
    else {
        status = cpu_scalar_add(a->data, scalar, output->data, a->size);
    }
#else
    status = cpu_scalar_add(a->data, scalar, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_identity(a->data, output->data, a->size);
    }
    else {
        status = cpu_identity(a->data, output->data, a->size);
    }
#else
    status = cpu_identity(a->data, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!deriv || !factor || deriv->device != factor->device || input_dim <= 0) return NULL;
    if(deriv->size % input_dim != 0 || factor->size != deriv->size / input_dim) return NULL;

    Tensor *output = tensor_create_device(deriv->shape, deriv->ndim, deriv->req_grad || factor->req_grad, deriv->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(deriv->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_scale_deriv(deriv->data, factor->data, input_dim, output->data, deriv->size);
    }
    else {
        status = cpu_scale_deriv(deriv->data, factor->data, input_dim, output->data, deriv->size);
    }
#else
    status = cpu_scale_deriv(deriv->data, factor->data, input_dim, output->data, deriv->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!d1 || !d2 || !f_prime || !f_double || input_dim <= 0) return NULL;
    if(d1->device != d2->device || d1->device != f_prime->device || d1->device != f_double->device) return NULL;
    if(d1->size != f_prime->size * input_dim || d2->size != f_prime->size * input_dim * input_dim || f_double->size != f_prime->size) return NULL;

    Tensor *output = tensor_create_device(d2->shape, d2->ndim, d1->req_grad || d2->req_grad || f_prime->req_grad || f_double->req_grad, d2->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(d1->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_chain_d2(d1->data, d2->data, f_prime->data, f_double->data, input_dim, f_prime->size, output->data);
    }
    else {
        status = cpu_chain_d2(d1->data, d2->data, f_prime->data, f_double->data, input_dim, f_prime->size, output->data);
    }
#else
    status = cpu_chain_d2(d1->data, d2->data, f_prime->data, f_double->data, input_dim, f_prime->size, output->data);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!deriv || !W || deriv->device != W->device || batch < 0 || in_features < 0 || out_features < 0 || input_dim <= 0) return NULL;
    if(deriv->size != batch * in_features * (order == 1 ? input_dim : input_dim * input_dim)) return NULL;
    if(W->size != in_features * out_features) return NULL;

    int channels = 1;
    for(int i = 0; i < order; i++){
        channels *= input_dim;
    }
    int output_shape[2] = {batch * out_features, channels};
    Tensor *output = tensor_create_device(output_shape, 2, deriv->req_grad || W->req_grad, deriv->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(deriv->device == DEVICE_CUDA && backend_cuda_available()){
        if(order == 1){
            status = cuda_deriv_matmult(deriv->data, W->data, batch, in_features, out_features, input_dim, output->data);
        }
        else {
            status = cuda_deriv2_matmult(deriv->data, W->data, batch, in_features, out_features, input_dim, output->data);
        }
    }
    else {
        if(order == 1){
            status = cpu_deriv_matmult(deriv->data, W->data, batch, in_features, out_features, input_dim, output->data);
        }
        else {
            status = cpu_deriv2_matmult(deriv->data, W->data, batch, in_features, out_features, input_dim, output->data);
        }
    }
#else
    if(order == 1){
        status = cpu_deriv_matmult(deriv->data, W->data, batch, in_features, out_features, input_dim, output->data);
    }
    else {
        status = cpu_deriv2_matmult(deriv->data, W->data, batch, in_features, out_features, input_dim, output->data);
    }
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);

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
    if(!d1 || input_dim <= 0 || component < 0 || component >= input_dim) return NULL;
    if(d1->size % input_dim != 0) return NULL;

    int out_dim = d1->shape[1] / input_dim;
    int shape[] = {d1->shape[0], out_dim};
    Tensor *output = tensor_create_device(shape, 2, d1->req_grad, d1->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(d1->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_select_d1(d1->data, input_dim, component, output->data, d1->size);
    }
    else {
        status = cpu_select_d1(d1->data, input_dim, component, output->data, d1->size);
    }
#else
    status = cpu_select_d1(d1->data, input_dim, component, output->data, d1->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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
    if(!d2 || input_dim <= 0 || p < 0 || p >= input_dim || q < 0 || q >= input_dim) return NULL;
    if(d2->size % (input_dim * input_dim) != 0) return NULL;

    int out_dim = d2->shape[1]  / (input_dim * input_dim);
    int shape[] = {d2->shape[0], out_dim};
    Tensor *output = tensor_create_device(shape, 2, d2->req_grad, d2->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(d2->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_select_d2(d2->data, input_dim, p, q, output->data, d2->size);
    }
    else {
        status = cpu_select_d2(d2->data, input_dim, p, q, output->data, d2->size);
    }
#else
    status = cpu_select_d2(d2->data, input_dim, p, q, output->data, d2->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
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

Tensor* tensor_select_col(Tensor *a, int component){
    if(!a || a->ndim != 2 || component < 0 || component >= a->shape[1]) return NULL;

    int shape[] = {a->shape[0], 1};
    Tensor *output = tensor_create_device(shape, 2, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_select_col(a->data, component, output->data, a->shape[0], a->shape[1]);
    }
    else {
        status = cpu_select_col(a->data, component, output->data, a->shape[0], a->shape[1]);
    }
#else
    status = cpu_select_col(a->data, component, output->data, a->shape[0], a->shape[1]);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = select_col_backward;
        SelectColCtx *ctx = malloc(sizeof(SelectColCtx));
        ctx->component = component;
        node->ctx = ctx;
        node->free_ctx = free;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_relu(Tensor *a){
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_relu(a->data, output->data, a->size);
    }
    else {
        status = cpu_relu(a->data, output->data, a->size);
    }
#else
    status = cpu_relu(a->data, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = relu_backward;
        node->ctx = NULL;
        node->free_ctx = NULL;
        output->grad_fn = node;
        tape_record_node(node);
    }
    return output;
}

Tensor* tensor_sigmoid(Tensor *a){
    if(!a) return NULL;

    Tensor *output = tensor_create_device(a->shape, a->ndim, a->req_grad, a->device);
    if(!output) return NULL;

    int status = 0;

#ifdef PINN_USE_CUDA
    if(a->device == DEVICE_CUDA && backend_cuda_available()){
        status = cuda_sigmoid(a->data, output->data, a->size);
    }
    else {
        status = cpu_sigmoid(a->data, output->data, a->size);
    }
#else
    status = cpu_sigmoid(a->data, output->data, a->size);
#endif

    if(status != 0){
        tensor_free(output);
        return NULL;
    }
    tape_record_tensor(output);
    if(output->req_grad){
        Node *node = malloc(sizeof(Node));
        node->inputs = malloc(sizeof(Tensor*));
        node->inputs[0] = a;
        node->n_inputs = 1;
        node->output = output;
        node->backward = sigmoid_backward;
        node->ctx = NULL;
        node->free_ctx = NULL;
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

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_mult_backward(a->data, b->data, output->grad, a->grad, b->grad, output->size, a->req_grad, b->req_grad);
        return;
    }
#endif

    cpu_mult_backward(a->data, b->data, output->grad, a->grad, b->grad, output->size, a->req_grad, b->req_grad);
}

static void add_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_add_backward(output->grad, a->grad, b->grad, output->size, a->req_grad, b->req_grad);
        return;
    }
#endif

    cpu_add_backward(output->grad, a->grad, b->grad, output->size, a->req_grad, b->req_grad);
}

static void sub_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_sub_backward(output->grad, a->grad, b->grad, output->size, a->req_grad, b->req_grad);
        return;
    }
#endif

    cpu_sub_backward(output->grad, a->grad, b->grad, output->size, a->req_grad, b->req_grad);
}

static void square_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_square_backward(a->data, output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif

    cpu_square_backward(a->data, output->grad, a->grad, a->size, a->req_grad);
}

static void mean_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_mean_backward(output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif

    cpu_mean_backward(output->grad, a->grad, a->size, a->req_grad);
}

static void tanh_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_tanh_backward(output->data, output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif

    cpu_tanh_backward(output->data, output->grad, a->grad, a->size, a->req_grad);
}

static void matmult_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *c = node->output;
    int rows = a->shape[0];
    int inner = a->shape[1];
    int cols = b->shape[1];

#ifdef PINN_USE_CUDA
    if(c->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_matmult_backward(a->data, b->data, c->grad, a->grad, b->grad, rows, inner, cols, a->req_grad, b->req_grad);
        return;
    }
#endif

    cpu_matmult_backward(a->data, b->data, c->grad, a->grad, b->grad, rows, inner, cols, a->req_grad, b->req_grad);
}


static void bias_add_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *b = node->inputs[1];
    Tensor *output = node->output;
    int rows = a->shape[0];
    int cols = a->shape[1];

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_bias_add_backward(output->grad, a->grad, b->grad, rows, cols, a->req_grad, b->req_grad);
        return;
    }
#endif

    cpu_bias_add_backward(output->grad, a->grad, b->grad, rows, cols, a->req_grad, b->req_grad);
}

static void scalar_mult_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    ScalarCtx *ctx = (ScalarCtx*)node->ctx;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_scalar_mult_backward(output->grad, ctx->scalar, a->grad, a->size, a->req_grad);
        return;
    }
#endif
    cpu_scalar_mult_backward(output->grad, ctx->scalar, a->grad, a->size, a->req_grad);
}

static void scalar_add_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_scalar_add_backward(output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif
    cpu_scalar_add_backward(output->grad, a->grad, a->size, a->req_grad);
}

static void identity_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_identity_backward(output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif

    cpu_identity_backward(output->grad, a->grad, a->size, a->req_grad);
}

static void scale_deriv_backward(Node *node){
    Tensor *deriv = node->inputs[0];
    Tensor *factor = node->inputs[1];
    Tensor *output = node->output;
    ScaleDerivCtx *ctx = (ScaleDerivCtx*)node->ctx;
    int input_dim = ctx->input_dim;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_scale_deriv_backward(deriv->data, factor->data, output->grad, deriv->grad, factor->grad, input_dim, deriv->size, deriv->req_grad, factor->req_grad);
        return;
    }
#endif
    cpu_scale_deriv_backward(deriv->data, factor->data, output->grad, deriv->grad, factor->grad, input_dim, deriv->size, deriv->req_grad, factor->req_grad);
}

static void chain_d2_backward(Node *node){
    Tensor *d1 = node->inputs[0];
    Tensor *d2 = node->inputs[1];
    Tensor *f_prime = node->inputs[2];
    Tensor *f_double = node->inputs[3];
    Tensor *output = node->output;
    ChainD2Ctx *ctx = (ChainD2Ctx*)node->ctx;
    int input_dim = ctx->input_dim;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_chain_d2_backward(d1->data, d2->data, f_prime->data, f_double->data, output->grad, d1->grad, d2->grad, f_prime->grad, f_double->grad, input_dim, f_prime->size, d1->req_grad, d2->req_grad, f_prime->req_grad, f_double->req_grad);
        return;
    }
#endif
    cpu_chain_d2_backward(d1->data, d2->data, f_prime->data, f_double->data, output->grad, d1->grad, d2->grad, f_prime->grad, f_double->grad, input_dim, f_prime->size, d1->req_grad, d2->req_grad, f_prime->req_grad, f_double->req_grad);
}

static void deriv_matmult_backward(Node *node){
    Tensor *deriv = node->inputs[0];
    Tensor *W = node->inputs[1];
    Tensor *output = node->output;
    DerivMatmultCtx *ctx = (DerivMatmultCtx*)node->ctx;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_deriv_matmult_backward(deriv->data, W->data, output->grad, deriv->grad, W->grad, ctx->batch, ctx->in_features, ctx->out_features, ctx->input_dim, ctx->order, deriv->req_grad, W->req_grad);
        return;
    }
#endif

    cpu_deriv_matmult_backward(deriv->data, W->data, output->grad, deriv->grad, W->grad, ctx->batch, ctx->in_features, ctx->out_features, ctx->input_dim, ctx->order, deriv->req_grad, W->req_grad);
}

static void select_d1_backward(Node *node){
    Tensor *d1 = node->inputs[0];
    Tensor *output = node->output;
    SelectD1Ctx *ctx = (SelectD1Ctx*)node->ctx;
    int input_dim = ctx->input_dim;
    int component = ctx->component;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_select_d1_backward(output->grad, d1->grad, input_dim, component, d1->size, d1->req_grad);
        return;
    }
#endif
    cpu_select_d1_backward(output->grad, d1->grad, input_dim, component, d1->size, d1->req_grad);
}

static void select_d2_backward(Node *node){
    Tensor *d2 = node->inputs[0];
    Tensor *output = node->output;
    SelectD2Ctx *ctx = (SelectD2Ctx*)node->ctx;
    int input_dim = ctx->input_dim;
    int p = ctx->p;
    int q = ctx->q;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_select_d2_backward(output->grad, d2->grad, input_dim, p, q, d2->size, d2->req_grad);
        return;
    }
#endif
    cpu_select_d2_backward(output->grad, d2->grad, input_dim, p, q, d2->size, d2->req_grad);
}

static void select_col_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;
    SelectColCtx *ctx = (SelectColCtx*)node->ctx;
    int component = ctx->component;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_select_col_backward(output->grad, a->grad, component, a->shape[0], a->shape[1], a->req_grad);
        return;
    }
#endif
    cpu_select_col_backward(output->grad, a->grad, component, a->shape[0], a->shape[1], a->req_grad);
}

static void relu_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_relu_backward(output->data, output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif
    cpu_relu_backward(output->data, output->grad, a->grad, a->size, a->req_grad);
}

static void sigmoid_backward(Node *node){
    Tensor *a = node->inputs[0];
    Tensor *output = node->output;

#ifdef PINN_USE_CUDA
    if(output->device == DEVICE_CUDA && backend_cuda_available()){
        cuda_sigmoid_backward(output->data, output->grad, a->grad, a->size, a->req_grad);
        return;
    }
#endif
    cpu_sigmoid_backward(output->data, output->grad, a->grad, a->size, a->req_grad);
}
