#include "pinn/surrogate/heat1d_inference.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
#include "pinn/nn/activations.h"
#include "pinn/nn/mlp.h"
#include "pinn/pinn/residual.h"
#include "pinn/surrogate/heat1d_modal_surrogate.h"
#include "pinn/surrogate/model_io.h"

struct Heat1DInference {
    MLP *mlp;
    float *model_input_lower;
    float *model_input_upper;
};

static int heat1d_model_contract_is_valid(const Heat1DInference *inference){
    if(!inference || !inference->mlp
        || !inference->model_input_lower || !inference->model_input_upper
        || inference->mlp->n_layers <= 0){
        return 0;
    }
    Linear *first = inference->mlp->layers[0];
    Linear *last = inference->mlp->layers[inference->mlp->n_layers - 1];
    if(!first || !last
        || first->input_dim != HEAT1D_MODAL_INPUT_DIM
        || last->output_dim != 1){
        return 0;
    }
    return isfinite(inference->model_input_lower[0])
        && isfinite(inference->model_input_upper[0])
        && fabsf(inference->model_input_lower[0] + 1.0f) <= 1e-5f
        && fabsf(inference->model_input_upper[0] - 1.0f) <= 1e-5f;
}

Heat1DInference *heat1d_inference_load(const char *model_path){
    if(!model_path) return NULL;
    Heat1DInference *inference = calloc(1, sizeof(Heat1DInference));
    if(!inference) return NULL;

    inference->mlp = pinn_model_load(
        model_path,
        &inference->model_input_lower,
        &inference->model_input_upper,
        tanh_activation
    );
    if(!heat1d_model_contract_is_valid(inference)){
        heat1d_inference_free(inference);
        return NULL;
    }
    return inference;
}

void heat1d_inference_free(Heat1DInference *inference){
    if(!inference) return;
    if(inference->mlp) mlp_free(inference->mlp);
    free(inference->model_input_lower);
    free(inference->model_input_upper);
    free(inference);
}

static int heat1d_physical_points_are_valid(
    const float *physical_points,
    int point_count
){
    for(int row = 0; row < point_count; row++){
        int offset = row * HEAT1D_INPUT_DIM;
        float t = physical_points[offset + HEAT1D_T];
        float x = physical_points[offset + HEAT1D_X];
        float alpha = physical_points[offset + HEAT1D_ALPHA];
        float a1 = physical_points[offset + HEAT1D_A1];
        float a2 = physical_points[offset + HEAT1D_A2];
        if(!isfinite(t) || !isfinite(x) || !isfinite(alpha)
            || !isfinite(a1) || !isfinite(a2)
            || t < 0.0f || t > 1.0f
            || x < 0.0f || x > 1.0f
            || alpha < 0.01f || alpha > 0.50f
            || a1 < -1.0f || a1 > 1.0f
            || a2 < -1.0f || a2 > 1.0f){
            return 0;
        }
    }
    return 1;
}

int heat1d_inference_predict(
    Heat1DInference *inference,
    const float *physical_points,
    int point_count,
    float *output
){
    if(!heat1d_model_contract_is_valid(inference)
        || !physical_points || !output || point_count <= 0
        || !heat1d_physical_points_are_valid(physical_points, point_count)){
        return 0;
    }

    int point_shape[2] = {point_count, HEAT1D_INPUT_DIM};
    Tensor *points = tensor_from_data(
        physical_points, point_shape, 2, 0
    );
    Tensor *tau1 = NULL;
    Tensor *tau2 = NULL;
    Tensor *input1 = NULL;
    Tensor *input2 = NULL;
    Tensor *prediction = NULL;
    Tape *tape = NULL;
    int success = 0;

    if(!points) goto cleanup;
    tau1 = heat1d_modal_tau_points(points, 1);
    tau2 = heat1d_modal_tau_points(points, 2);
    input1 = heat1d_modal_normalize_tau(tau1);
    input2 = heat1d_modal_normalize_tau(tau2);
    if(!tau1 || !tau2 || !input1 || !input2) goto cleanup;

    tape = tape_create();
    if(!tape) goto cleanup;
    set_curr_tape(tape);

    Tensor *network1 = mlp_forward(inference->mlp, input1);
    Tensor *network2 = mlp_forward(inference->mlp, input2);
    Tensor *q1 = heat1d_modal_coefficient(network1, tau1);
    Tensor *q2 = heat1d_modal_coefficient(network2, tau2);
    prediction = heat1d_modal_reconstruct(q1, q2, points);
    if(!network1 || !network2 || !q1 || !q2 || !prediction) goto cleanup;

    memcpy(output, prediction->data, (size_t)point_count * sizeof(float));
    success = 1;

cleanup:
    set_curr_tape(NULL);
    tape_free(tape);
    tensor_free(input1);
    tensor_free(input2);
    tensor_free(tau1);
    tensor_free(tau2);
    tensor_free(points);
    return success;
}

int heat1d_inference_input_dim(void){
    return HEAT1D_INPUT_DIM;
}
