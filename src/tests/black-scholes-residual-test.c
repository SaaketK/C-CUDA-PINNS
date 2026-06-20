/*
 * src/tests/black-scholes-residual-test.c
 *
 * Smoke tests for the differentiable 1D Black-Scholes residual.
 */

#include <math.h>
#include <stdio.h>
#include "pinn/autodiff/jet.h"
#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
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

static int test_black_scholes_residual_quadratic(void){
    int point_shape[2] = {2, 2};
    float point_values[4] = {
        0.25f, 2.0f,
        0.75f, 3.0f
    };
    Tensor *points = tensor_from_data(point_values, point_shape, 2, 0);

    int value_shape[2] = {2, 1};
    float value_values[2] = {
        0.25f + 2.0f + 2.0f * 2.0f,
        0.75f + 3.0f + 3.0f * 3.0f
    };
    Tensor *value = tensor_from_data(value_values, value_shape, 2, 0);
    int d1_shape[2] = {2, 2};
    int d2_shape[2] = {2, 4};
    Tensor *d1 = tensor_create(d1_shape, 2, 0);
    Tensor *d2 = tensor_create(d2_shape, 2, 0);
    JetTensor *V = jet_from_parts(value, d1, d2, 2);

    for(int i = 0; i < point_shape[0]; i++){
        float S = points->data[i * 2 + 1];
        jet_set_d1(V, i, 0, 1.0f);
        jet_set_d1(V, i, 1, 1.0f + 2.0f * S);
        jet_set_d2(V, i, 0, 0, 0.0f);
        jet_set_d2(V, i, 0, 1, 0.0f);
        jet_set_d2(V, i, 1, 0, 0.0f);
        jet_set_d2(V, i, 1, 1, 2.0f);
    }

    BlackScholes1DParams params = {
        .sigma = 0.2f,
        .r = 0.05f,
        .K = 1.0f,
        .T = 1.0f
    };

    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *residual = black_scholes1d_residual(V, points, &params);

    int ok = 1;
    for(int i = 0; i < point_shape[0]; i++){
        float t = points->data[i * 2];
        float S = points->data[i * 2 + 1];
        float expected = 1.0f + (params.sigma * params.sigma + params.r) * S * S - params.r * t;
        char name[64];
        snprintf(name, sizeof(name), "black-scholes residual sample %d", i);
        ok = check_close(name, residual->data[i], expected, 1e-5f) && ok;
    }

    tape_free(tape);
    jet_free(V);
    tensor_free(points);
    return ok;
}

int main(void){
    if(!test_black_scholes_residual_quadratic()){
        printf("black-scholes residual tests failed\n");
        return 1;
    }

    printf("all black-scholes residual tests passed\n");
    return 0;
}
