/* Regression tests for the parametric Heat 1D surrogate. */

#include <math.h>
#include <stdio.h>

#include "pinn/autodiff/jet.h"
#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
#include "pinn/nn/mlp.h"
#include "pinn/pinn/residual.h"
#include "pinn/surrogate/heat1d_surrogate.h"
#include "pinn/surrogate/heat1d_modal_surrogate.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int check_close(const char *name, float actual, float expected, float tol){
    float error = fabsf(actual - expected);
    if(!isfinite(actual) || error > tol){
        printf("%s FAILED: actual=%g expected=%g abs_error=%g\n",
            name, actual, expected, error);
        return 0;
    }
    return 1;
}

static float initial_condition(float x, float a1, float a2){
    return a1 * sinf((float)M_PI * x)
        + a2 * sinf(2.0f * (float)M_PI * x);
}

static float initial_condition_xx(float x, float a1, float a2){
    float pi2 = (float)M_PI * (float)M_PI;
    return -pi2 * (a1 * sinf((float)M_PI * x)
        + 4.0f * a2 * sinf(2.0f * (float)M_PI * x));
}

static int test_parametric_ansatz_constraints(void){
    const int n = 6;
    int shape[2] = {n, HEAT1D_INPUT_DIM};
    float point_values[] = {
        0.0f, 0.20f, 0.05f,  0.70f, -0.40f,
        0.0f, 0.80f, 0.20f, -0.60f,  0.25f,
        0.35f, 0.0f, 0.40f,  0.20f,  0.30f,
        0.60f, 1.0f, 0.10f, -0.80f,  0.90f,
        0.45f, 0.33f, 0.05f, 1.00f, -1.00f,
        0.75f, 0.67f, 0.48f, -0.30f, 0.60f,
    };
    float raw_values[] = {3.0f, -2.0f, 7.0f, -4.0f, 0.5f, -1.5f};

    Tensor *points = tensor_from_data(point_values, shape, 2, 0);
    int raw_shape[2] = {n, 1};
    Tensor *raw = tensor_from_data(raw_values, raw_shape, 2, 0);
    Tensor *u = heat1d_parametric_ansatz(raw, points);
    int ok = 1;

    for(int i = 0; i < n; i++){
        float t = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_T];
        float x = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_X];
        float a1 = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_A1];
        float a2 = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_A2];
        float expected = (1.0f - t) * initial_condition(x, a1, a2)
            + t * x * (1.0f - x) * raw_values[i];
        ok = check_close("parametric ansatz", u->data[i], expected, 1e-5f) && ok;
    }

    tensor_free(u);
    tensor_free(raw);
    tensor_free(points);
    return ok;
}

static int test_parametric_zero_network_residual(void){
    const int n = 4;
    int shape[2] = {n, HEAT1D_INPUT_DIM};
    float point_values[] = {
        0.15f, 0.20f, 0.05f,  0.70f, -0.40f,
        0.35f, 0.45f, 0.20f, -0.60f,  0.25f,
        0.55f, 0.70f, 0.40f,  0.20f,  0.30f,
        0.80f, 0.35f, 0.48f, -0.80f,  0.90f,
    };

    Tensor *points = tensor_from_data(point_values, shape, 2, 0);
    JetTensor *input = jet_create_input(points, HEAT1D_INPUT_DIM);
    int sizes[2] = {HEAT1D_INPUT_DIM, 1};
    MLP *mlp = mlp_create(sizes, 2, NULL);
    for(int i = 0; i < mlp->layers[0]->W->size; i++){
        mlp->layers[0]->W->data[i] = 0.0f;
    }
    mlp->layers[0]->b->data[0] = 0.0f;

    JetTensor *network = jet_mlp_forward(mlp, input);
    Tensor *residual = heat1d_parametric_ansatz_residual(network, points);
    int ok = 1;

    for(int i = 0; i < n; i++){
        float t = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_T];
        float x = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_X];
        float alpha = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_ALPHA];
        float a1 = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_A1];
        float a2 = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_A2];
        float expected = -initial_condition(x, a1, a2)
            - alpha * (1.0f - t) * initial_condition_xx(x, a1, a2);
        ok = check_close("parametric residual", residual->data[i], expected, 1e-5f) && ok;
    }

    tensor_free(residual);
    jet_free(network);
    jet_free(input);
    mlp_free(mlp);
    return ok;
}

static int test_input_normalization(void){
    int shape[2] = {3, HEAT1D_INPUT_DIM};
    float raw_values[] = {
        0.0f, 0.0f, 0.01f, -1.0f,  1.0f,
        0.5f, 0.5f, 0.255f, 0.0f, -0.5f,
        1.0f, 1.0f, 0.50f, 1.0f, -1.0f,
    };
    float expected[] = {
        -1.0f, -1.0f, -1.0f, -1.0f,  1.0f,
         0.0f,  0.0f,  0.0f,  0.0f, -0.5f,
         1.0f,  1.0f,  1.0f,  1.0f, -1.0f,
    };
    Tensor *raw = tensor_from_data(raw_values, shape, 2, 0);
    Tensor *normalized = heat1d_normalize_inputs(raw);
    int ok = normalized != NULL;
    if(ok){
        for(int i = 0; i < normalized->size; i++){
            ok = check_close("normalized surrogate input", normalized->data[i], expected[i], 1e-5f) && ok;
        }
    }
    tensor_free(normalized);
    tensor_free(raw);
    return ok;
}

static int test_coefficient_fourier_features(void){
    int shape[2] = {1, HEAT1D_INPUT_DIM};
    float raw_values[] = {0.5f, 0.25f, 0.255f, 1.0f, 0.0f};
    Tensor *raw = tensor_from_data(raw_values, shape, 2, 0);
    Tensor *features = heat1d_coefficient_features(raw);
    JetTensor *jet = heat1d_coefficient_feature_jet_input(raw);
    float root_half = sqrtf(0.5f);
    float pi = (float)M_PI;
    int ok = features != NULL && jet != NULL;

    if(ok){
        ok = check_close("feature t", features->data[HEAT1D_COEFF_T], 0.0f, 1e-5f) && ok;
        ok = check_close("feature x", features->data[HEAT1D_COEFF_X], -0.5f, 1e-5f) && ok;
        ok = check_close("feature alpha", features->data[HEAT1D_COEFF_ALPHA], 0.0f, 1e-5f) && ok;
        ok = check_close("feature sin pi x", features->data[HEAT1D_COEFF_SIN_PI_X], root_half, 1e-5f) && ok;
        ok = check_close("feature cos pi x", features->data[HEAT1D_COEFF_COS_PI_X], root_half, 1e-5f) && ok;
        ok = check_close("feature sin 2pi x", features->data[HEAT1D_COEFF_SIN_2PI_X], 1.0f, 1e-5f) && ok;
        ok = check_close("feature cos 2pi x", features->data[HEAT1D_COEFF_COS_2PI_X], 0.0f, 1e-5f) && ok;
        ok = check_close("feature dt", jet_get_d1(jet, HEAT1D_COEFF_T, HEAT1D_T), 2.0f, 1e-5f) && ok;
        ok = check_close("feature dx", jet_get_d1(jet, HEAT1D_COEFF_X, HEAT1D_X), 2.0f, 1e-5f) && ok;
        ok = check_close("feature dalpha", jet_get_d1(jet, HEAT1D_COEFF_ALPHA, HEAT1D_ALPHA), 2.0f / 0.49f, 1e-5f) && ok;
        ok = check_close("sin pi x dx", jet_get_d1(jet, HEAT1D_COEFF_SIN_PI_X, HEAT1D_X), pi * root_half, 1e-5f) && ok;
        ok = check_close("sin 2pi x dx", jet_get_d1(jet, HEAT1D_COEFF_SIN_2PI_X, HEAT1D_X), 0.0f, 1e-5f) && ok;
        ok = check_close("sin 2pi x dxx", jet_get_d2(jet, HEAT1D_COEFF_SIN_2PI_X, HEAT1D_X, HEAT1D_X), -4.0f * pi * pi, 1e-4f) && ok;
    }

    jet_free(jet);
    tensor_free(features);
    tensor_free(raw);
    return ok;
}

static int test_modal_coordinate_and_initial_condition(void){
    int shape[2] = {3, HEAT1D_MODAL_INPUT_DIM};
    float tau_values[] = {
        0.0f,
        0.5f * HEAT1D_MODAL_TAU_MAX,
        HEAT1D_MODAL_TAU_MAX,
    };
    float network_values[] = {7.0f, -0.25f, 0.5f};
    Tensor *tau = tensor_from_data(tau_values, shape, 2, 0);
    Tensor *normalized = heat1d_modal_normalize_tau(tau);
    JetTensor *input = heat1d_modal_jet_input(tau);
    Tensor *network = tensor_from_data(network_values, shape, 2, 0);
    Tensor *q = heat1d_modal_coefficient(network, tau);
    int ok = normalized && input && q;

    if(ok){
        ok = check_close("modal tau lower", normalized->data[0], -1.0f, 1e-6f) && ok;
        ok = check_close("modal tau midpoint", normalized->data[1], 0.0f, 1e-6f) && ok;
        ok = check_close("modal tau upper", normalized->data[2], 1.0f, 1e-6f) && ok;
        ok = check_close(
            "modal normalized derivative",
            jet_get_d1(input, 0, HEAT1D_MODAL_TAU),
            2.0f / HEAT1D_MODAL_TAU_MAX,
            1e-6f
        ) && ok;
        ok = check_close("modal hard initial condition", q->data[0], 1.0f, 1e-6f) && ok;
    }

    tensor_free(q);
    tensor_free(network);
    jet_free(input);
    tensor_free(normalized);
    tensor_free(tau);
    return ok;
}

static int test_modal_residual(void){
    int shape[2] = {3, HEAT1D_MODAL_INPUT_DIM};
    float tau_values[] = {0.0f, 0.5f, 2.0f};
    Tensor *tau = tensor_from_data(tau_values, shape, 2, 0);
    JetTensor *input = heat1d_modal_jet_input(tau);
    int sizes[] = {HEAT1D_MODAL_INPUT_DIM, 1};
    MLP *mlp = mlp_create(sizes, 2, NULL);
    mlp->layers[0]->W->data[0] = 0.0f;
    mlp->layers[0]->b->data[0] = 0.0f;

    Tape *tape = tape_create();
    set_curr_tape(tape);
    JetTape *jet_tape = jet_tape_create();
    set_curr_jet_tape(jet_tape);
    JetTensor *network = jet_mlp_forward(mlp, input);
    Tensor *residual = heat1d_modal_residual(network, tau);
    int ok = residual != NULL;
    if(ok){
        for(int i = 0; i < 3; i++){
            ok = check_close(
                "modal ODE residual",
                residual->data[i],
                tau_values[i] / ((1.0f + tau_values[i]) * (1.0f + tau_values[i])),
                1e-5f
            ) && ok;
        }
    }

    jet_tape_free_shallow(jet_tape);
    jet_free(input);
    tape_free(tape);
    tensor_free(tau);
    mlp_free(mlp);
    return ok;
}

static int test_modal_reconstruction(void){
    const int n = 3;
    int point_shape[2] = {n, HEAT1D_INPUT_DIM};
    float point_values[] = {
        0.0f, 0.25f, 0.01f,  0.7f, -0.4f,
        0.4f, 0.50f, 0.20f, -0.6f,  0.3f,
        0.9f, 0.75f, 0.50f,  0.2f,  0.9f,
    };
    Tensor *points = tensor_from_data(point_values, point_shape, 2, 0);
    Tensor *tau1 = heat1d_modal_tau_points(points, 1);
    Tensor *tau2 = heat1d_modal_tau_points(points, 2);
    int coefficient_shape[2] = {n, 1};
    float q1_values[n];
    float q2_values[n];
    for(int i = 0; i < n; i++){
        q1_values[i] = expf(-tau1->data[i]);
        q2_values[i] = expf(-tau2->data[i]);
    }
    Tensor *q1 = tensor_from_data(q1_values, coefficient_shape, 2, 0);
    Tensor *q2 = tensor_from_data(q2_values, coefficient_shape, 2, 0);
    Tensor *prediction = heat1d_modal_reconstruct(q1, q2, points);
    int ok = prediction != NULL;
    if(ok){
        for(int i = 0; i < n; i++){
            float t = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_T];
            float x = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_X];
            float alpha = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_ALPHA];
            float a1 = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_A1];
            float a2 = point_values[i * HEAT1D_INPUT_DIM + HEAT1D_A2];
            float expected = a1 * sinf((float)M_PI * x)
                * expf(-alpha * (float)M_PI * (float)M_PI * t)
                + a2 * sinf(2.0f * (float)M_PI * x)
                * expf(-4.0f * alpha * (float)M_PI * (float)M_PI * t);
            ok = check_close(
                "modal reconstruction", prediction->data[i], expected, 1e-5f
            ) && ok;
        }
    }

    tensor_free(prediction);
    tensor_free(q1);
    tensor_free(q2);
    tensor_free(tau1);
    tensor_free(tau2);
    tensor_free(points);
    return ok;
}

int main(void){
    if(!test_parametric_ansatz_constraints()
        || !test_parametric_zero_network_residual()
        || !test_input_normalization()
        || !test_coefficient_fourier_features()
        || !test_modal_coordinate_and_initial_condition()
        || !test_modal_residual()
        || !test_modal_reconstruction()){
        return 1;
    }

    printf("all parametric Heat 1D tests passed\n");
    return 0;
}
