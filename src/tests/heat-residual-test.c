/*
 * src/tests/heat-residual-test.c
 *
 * Smoke tests for extracting PDE derivatives from JetTensor MLP outputs and
 * forming a simple 1D heat-equation residual.
 */

#include <math.h>
#include <stdio.h>
#include "pinn/autodiff/jet.h"
#include "pinn/core/tensor.h"
#include "pinn/nn/mlp.h"

static int check_close(const char *name, float actual, float expected, float tol){
    float err = fabsf(actual - expected);
    printf("%s actual=%f expected=%f abs_err=%f\n", name, actual, expected, err);
    if(err > tol){
        printf("%s FAILED\n", name);
        return 0;
    }
    return 1;
}

static float heat_residual_1d(JetTensor *u, int sample_index, float alpha){
    float u_t = jet_get_d1(u, sample_index, 0);
    float u_xx = jet_get_d2(u, sample_index, 1, 1);
    return u_t - alpha * u_xx;
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
    float alpha = 0.5f;
    int ok = 1;

    ok = check_close("heat u sample 0", u->value->data[0], 24.0f, 1e-6f) && ok;
    ok = check_close("heat u_t sample 0", jet_get_d1(u, 0, 0), 2.0f, 1e-6f) && ok;
    ok = check_close("heat u_x sample 0", jet_get_d1(u, 0, 1), 3.0f, 1e-6f) && ok;
    ok = check_close("heat u_xx sample 0", jet_get_d2(u, 0, 1, 1), 0.0f, 1e-6f) && ok;
    ok = check_close("heat residual sample 0", heat_residual_1d(u, 0, alpha), 2.0f, 1e-6f) && ok;

    ok = check_close("heat u sample 1", u->value->data[1], -2.0f, 1e-6f) && ok;
    ok = check_close("heat residual sample 1", heat_residual_1d(u, 1, alpha), 2.0f, 1e-6f) && ok;

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

    printf("all heat residual tests passed\n");
    return 0;
}
