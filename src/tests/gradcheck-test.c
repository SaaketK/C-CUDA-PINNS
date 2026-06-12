/*
 * src/tests/gradcheck-test.c
 *
 * Finite-difference checks for reverse-mode gradients.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/core/autograd.h"
#include "pinn/nn/mlp.h"

static int check_close(const char *name, float autograd, float numerical, float tol){
    float abs_err = fabsf(autograd - numerical);
    printf("%s autograd=%f numerical=%f abs_err=%f\n", name, autograd, numerical, abs_err);
    if(abs_err > tol){
        printf("%s FAILED\n", name);
        return 0;
    }
    printf("%s passed\n", name);
    return 1;
}

static float square_loss_value(Tensor *x){
    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *sq = tensor_square(x);
    Tensor *loss = tensor_mean(sq);
    float value = loss->data[0];
    tape_free(tape);
    return value;
}

static int gradcheck_square(void){
    int shape[1] = {3};
    float values[3] = {1.25f, -0.75f, 0.5f};
    Tensor *x = tensor_from_data(values, shape, 1, 1);
    int index = 1;
    float eps = 1e-3f;
    float original = x->data[index];

    Tape *tape = tape_create();
    set_curr_tape(tape);
    tensor_zero_grad(x);
    Tensor *sq = tensor_square(x);
    Tensor *loss = tensor_mean(sq);
    backward(loss);
    float autograd = x->grad[index];
    tape_free(tape);

    x->data[index] = original + eps;
    float plus = square_loss_value(x);
    x->data[index] = original - eps;
    float minus = square_loss_value(x);
    x->data[index] = original;

    float numerical = (plus - minus) / (2.0f * eps);
    int ok = check_close("square gradcheck", autograd, numerical, 1e-3f);
    tensor_free(x);
    return ok;
}

static float linear_loss_value(Linear *linear, Tensor *x){
    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *y = linear_forward(linear, x);
    Tensor *loss = tensor_mean(y);
    float value = loss->data[0];
    tape_free(tape);
    return value;
}

static int gradcheck_linear(void){
    int x_shape[2] = {2, 3};
    float x_values[6] = {
        0.5f, -1.0f, 2.0f,
        1.5f, 0.25f, -0.5f
    };
    Tensor *x = tensor_from_data(x_values, x_shape, 2, 0);
    Linear *linear = linear_create(3, 2);
    for(int i = 0; i < linear->W->size; i++){
        linear->W->data[i] = 0.1f * (float)(i + 1);
    }
    tensor_zero(linear->b);

    int index = 3;
    float eps = 1e-3f;
    float original = linear->W->data[index];

    Tape *tape = tape_create();
    set_curr_tape(tape);
    tensor_zero_grad(linear->W);
    tensor_zero_grad(linear->b);
    Tensor *y = linear_forward(linear, x);
    Tensor *loss = tensor_mean(y);
    backward(loss);
    float autograd = linear->W->grad[index];
    tape_free(tape);

    linear->W->data[index] = original + eps;
    float plus = linear_loss_value(linear, x);
    linear->W->data[index] = original - eps;
    float minus = linear_loss_value(linear, x);
    linear->W->data[index] = original;

    float numerical = (plus - minus) / (2.0f * eps);
    int ok = check_close("linear W gradcheck", autograd, numerical, 1e-3f);
    linear_free(linear);
    tensor_free(x);
    return ok;
}

static void set_mlp_deterministic(MLP *mlp){
    float value = -0.2f;
    for(int layer = 0; layer < mlp->n_layers; layer++){
        for(int i = 0; i < mlp->layers[layer]->W->size; i++){
            mlp->layers[layer]->W->data[i] = value;
            value += 0.07f;
        }
        tensor_zero(mlp->layers[layer]->b);
    }
}

static float mlp_loss_value(MLP *mlp, Tensor *x, Tensor *target){
    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *pred = mlp_forward(mlp, x);
    Tensor *loss = tensor_mse(pred, target);
    float value = loss->data[0];
    tape_free(tape);
    return value;
}

static int gradcheck_mlp(void){
    int x_shape[2] = {2, 2};
    int y_shape[2] = {2, 1};
    float x_values[4] = {
        -0.5f, 0.25f,
        0.75f, -1.0f
    };
    float y_values[2] = {0.3f, -0.2f};
    Tensor *x = tensor_from_data(x_values, x_shape, 2, 0);
    Tensor *target = tensor_from_data(y_values, y_shape, 2, 0);
    int sizes[3] = {2, 3, 1};
    MLP *mlp = mlp_create(sizes, 3);
    set_mlp_deterministic(mlp);

    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    int param_index = 0;
    int data_index = 2;
    float eps = 1e-3f;
    float original = params[param_index]->data[data_index];

    Tape *tape = tape_create();
    set_curr_tape(tape);
    for(int i = 0; i < n_params; i++){
        tensor_zero_grad(params[i]);
    }
    Tensor *pred = mlp_forward(mlp, x);
    Tensor *loss = tensor_mse(pred, target);
    backward(loss);
    float autograd = params[param_index]->grad[data_index];
    tape_free(tape);

    params[param_index]->data[data_index] = original + eps;
    float plus = mlp_loss_value(mlp, x, target);
    params[param_index]->data[data_index] = original - eps;
    float minus = mlp_loss_value(mlp, x, target);
    params[param_index]->data[data_index] = original;

    float numerical = (plus - minus) / (2.0f * eps);
    int ok = check_close("mlp param gradcheck", autograd, numerical, 1e-3f);

    free(params);
    mlp_free(mlp);
    tensor_free(target);
    tensor_free(x);
    return ok;
}

int main(void){
    int ok = 1;
    ok = gradcheck_square() && ok;
    ok = gradcheck_linear() && ok;
    ok = gradcheck_mlp() && ok;

    if(!ok){
        printf("gradient checks failed\n");
        return 1;
    }

    printf("all gradient checks passed\n");
    return 0;
}
