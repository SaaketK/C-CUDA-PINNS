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

static float stable_sigmoid(float z){
    if(z >= 0.0f){
        float e = expf(-z);
        return 1.0f / (1.0f + e);
    }

    float e = expf(z);
    return e / (1.0f + e);
}

static float stable_softplus(float z){
    if(z > 20.0f){
        return z;
    }
    if(z < -20.0f){
        return expf(z);
    }
    return log1pf(expf(z));
}

static void smooth_payoff(float S, BlackScholes1DParams *params, float *payoff, float *payoff_S, float *payoff_SS){
    float beta = params->payoff_beta;
    float z = beta * (S - params->K);
    float sig = stable_sigmoid(z);
    *payoff = stable_softplus(z) / beta;
    *payoff_S = sig;
    *payoff_SS = beta * sig * (1.0f - sig);
}

static int test_black_scholes_residual_quadratic(void){
    int point_shape[2] = {2, 2};
    float point_values[4] = {
        0.25f, 0.5f,
        0.75f, 1.5f
    };
    Tensor *points = tensor_from_data(point_values, point_shape, 2, 0);

    int value_shape[2] = {2, 1};
    float value_values[2] = {
        0.25f + 0.5f + 0.5f * 0.5f,
        0.75f + 1.5f + 1.5f * 1.5f
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
        .T = 1.0f,
        .S_max = 2.0f,
        .payoff_beta = 0.25f
    };

    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *residual = black_scholes1d_residual(V, points, &params);

    int ok = 1;
    for(int i = 0; i < point_shape[0]; i++){
        float t = points->data[i * 2];
        float S = points->data[i * 2 + 1];
        float tau = params.T - t;
        float s = S / params.S_max;
        float one_minus_s = 1.0f - s;
        float exp_term = expf(-params.r * tau);
        float N = t + S + S * S;
        float payoff;
        float payoff_S;
        float payoff_SS;
        float payoff_max;
        float payoff_max_S;
        float payoff_max_SS;
        smooth_payoff(S, &params, &payoff, &payoff_S, &payoff_SS);
        smooth_payoff(params.S_max, &params, &payoff_max, &payoff_max_S, &payoff_max_SS);
        float right = params.S_max - params.K * exp_term;
        float C = right - payoff_max;
        float C_t = -params.r * params.K * exp_term;
        float A = payoff + s * C;
        float A_t = s * C_t;
        float A_S = payoff_S + C / params.S_max;
        float A_SS = payoff_SS;
        float B = tau * s * one_minus_s;
        float B_t = -s * one_minus_s;
        float B_S = tau * (1.0f - 2.0f * s) / params.S_max;
        float B_SS = -2.0f * tau / (params.S_max * params.S_max);
        float N_t = 1.0f;
        float N_S = 1.0f + 2.0f * S;
        float N_SS = 2.0f;
        float V_ansatz = A + B * N;
        float V_t = A_t + B_t * N + B * N_t;
        float V_S = A_S + B_S * N + B * N_S;
        float V_SS = A_SS + B_SS * N + 2.0f * B_S * N_S + B * N_SS;
        float expected = V_t
            + 0.5f * params.sigma * params.sigma * S * S * V_SS
            + params.r * S * V_S
            - params.r * V_ansatz;
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
