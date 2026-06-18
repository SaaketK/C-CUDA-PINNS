/*
 * src/tests/heat-residual-test.c
 *
 * Smoke tests for extracting PDE derivatives from JetTensor MLP outputs and
 * forming a simple 1D heat-equation residual.
 */

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "pinn/autodiff/jet.h"
#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
#include "pinn/nn/mlp.h"
#include "pinn/pinn/residual.h"

static int check_close(const char *name, float actual, float expected, float tol){
    float err = fabsf(actual - expected);
    printf("%s actual=%f expected=%f abs_err=%f\n", name, actual, expected, err);
    if(err > tol){
        printf("%s FAILED\n", name);
        return 0;
    }
    return 1;
}

static int test_linear_heat_residual(void){
    int input_shape[2] = {2, 2};
    float input_values[4] = {
        4.0f, 5.0f,
        1.5f, -2.0f
    };
    Tensor *input = tensor_from_data(input_values, input_shape, 2, 0);
    JetTensor *jet_input = jet_create_input(input, 2);

    int sizes[2] = {2, 1};
    MLP *mlp = mlp_create(sizes, 2);
    mlp->layers[0]->W->data[0] = 2.0f;
    mlp->layers[0]->W->data[1] = 3.0f;
    mlp->layers[0]->b->data[0] = 1.0f;

    JetTensor *u = jet_mlp_forward(mlp, jet_input);
    Heat1DParams params = {.alpha = 0.5f};
    Tensor *residual = heat1d_residual(u, input, &params);
    int ok = 1;

    ok = check_close("heat u sample 0", u->value->data[0], 24.0f, 1e-6f) && ok;
    ok = check_close("heat u_t sample 0", jet_get_d1(u, 0, 0), 2.0f, 1e-6f) && ok;
    ok = check_close("heat u_x sample 0", jet_get_d1(u, 0, 1), 3.0f, 1e-6f) && ok;
    ok = check_close("heat u_xx sample 0", jet_get_d2(u, 0, 1, 1), 0.0f, 1e-6f) && ok;
    ok = check_close("heat residual sample 0", residual->data[0], 2.0f, 1e-6f) && ok;

    ok = check_close("heat u sample 1", u->value->data[1], -2.0f, 1e-6f) && ok;
    ok = check_close("heat residual sample 1", residual->data[1], 2.0f, 1e-6f) && ok;

    tensor_free(residual);
    jet_free(u);
    jet_free(jet_input);
    mlp_free(mlp);
    return ok;
}

static int test_heat_physics_loss_backprop(void){
    int input_shape[2] = {2, 2};
    float input_values[4] = {
        0.2f, 0.3f,
        0.7f, 0.8f
    };
    Tensor *input = tensor_from_data(input_values, input_shape, 2, 0);
    JetTensor *jet_input = jet_create_input(input, 2);

    int sizes[3] = {2, 3, 1};
    MLP *mlp = mlp_create(sizes, 3);
    float w0_values[6] = {
        0.4f, -0.2f, 0.7f,
        0.3f, 0.5f, -0.6f
    };
    float b0_values[3] = {0.1f, -0.3f, 0.2f};
    float w1_values[3] = {0.8f, -0.5f, 0.4f};
    for(int i = 0; i < 6; i++){
        mlp->layers[0]->W->data[i] = w0_values[i];
    }
    for(int i = 0; i < 3; i++){
        mlp->layers[0]->b->data[i] = b0_values[i];
        mlp->layers[1]->W->data[i] = w1_values[i];
    }
    mlp->layers[1]->b->data[0] = 0.05f;

    JetTensor *u = jet_mlp_forward(mlp, jet_input);
    Heat1DParams params = {.alpha = 0.5f};
    Tensor *residual = heat1d_residual(u, input, &params);
    Tensor *loss = residual_mse_loss(residual);
    backward(loss);

    int n_params = 0;
    Tensor **params_list = mlp_parameters(mlp, &n_params);
    float max_abs_grad = 0.0f;
    for(int i = 0; i < n_params; i++){
        Tensor *param = params_list[i];
        for(int j = 0; j < param->size; j++){
            float abs_grad = fabsf(param->grad[j]);
            if(abs_grad > max_abs_grad){
                max_abs_grad = abs_grad;
            }
        }
    }
    free(params_list);

    int ok = max_abs_grad > 1e-7f;
    printf("heat physics loss max abs param grad=%f\n", max_abs_grad);
    if(!ok){
        printf("heat physics loss backprop FAILED\n");
    }

    tensor_free(loss);
    tensor_free(residual);
    jet_free(u);
    jet_free(jet_input);
    mlp_free(mlp);
    return ok;
}

int main(void){
    if(!test_linear_heat_residual()){
        printf("heat residual tests failed\n");
        return 1;
    }
    if(!test_heat_physics_loss_backprop()){
        printf("heat residual tests failed\n");
        return 1;
    }

    printf("all heat residual tests passed\n");
    return 0;
}
