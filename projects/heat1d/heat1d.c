/*
 * src/main.c
 *
 * Placeholder executable for the framework. 
 * Smoke tests live under src/tests.
 * cd /Users/saaketk/PycharmProjects/PINN && cmake --build C-CUDA-PINNs/build --target pinn_main && (cd C-CUDA-PINNs && ./build/pinn_main) && MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python C-CUDA-PINNs/projects/heat1d/plot_heat1d.py
*/

#define _POSIX_C_SOURCE 199309L

#include "pinn/core/autograd.h"
#include "pinn/core/backend.h"
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/nn/mlp.h"
#include "pinn/nn/optimizer.h"
#include "pinn/nn/lbfgs.h"
#include "pinn/pinn/residual.h"
#include "pinn/pinn/sampler.h"
#include "pinn/pinn/trainer.h"
#include "pinn/autodiff/jet.h"
#include "pinn/surrogate/model_io.h"
#include "pinn/surrogate/heat1d_surrogate.h"
#include "pinn/surrogate/heat1d_modal_surrogate.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static float heat1d_exact(float t, float x, float alpha, float a1, float a2){
    float pi2 = (float)M_PI * (float)M_PI;
    return a1 * expf(-alpha * pi2 * t) * sinf((float)M_PI * x)
        + a2 * expf(-4.0f * alpha * pi2 * t)
            * sinf(2.0f * (float)M_PI * x);
}

static double elapsed_seconds(struct timespec start, struct timespec end){
    return (double)(end.tv_sec - start.tv_sec) + (double)(end.tv_nsec - start.tv_nsec) / 1e9;
}

typedef struct {
    float alpha;
    float a1;
    float a2;
} Heat1DValidationCase;

static const Heat1DValidationCase VALIDATION_CASES[] = {
    {0.10f,  1.00f,  0.00f}, /* original fixed Heat 1D baseline */
    {0.05f,  0.70f, -0.40f},
    {0.20f, -0.60f,  0.25f},
    {0.45f,  0.25f,  0.90f},
};
static const Heat1DValidationCase MODE1_VALIDATION_CASES[] = {
    {0.01f, 1.0f, 0.0f}, {0.05f, 1.0f, 0.0f},
    {0.10f, 1.0f, 0.0f}, {0.20f, 1.0f, 0.0f},
    {0.35f, 1.0f, 0.0f}, {0.50f, 1.0f, 0.0f},
};
static const Heat1DValidationCase MODE2_VALIDATION_CASES[] = {
    {0.01f, 0.0f, 1.0f}, {0.05f, 0.0f, 1.0f},
    {0.10f, 0.0f, 1.0f}, {0.20f, 0.0f, 1.0f},
    {0.35f, 0.0f, 1.0f}, {0.50f, 0.0f, 1.0f},
};
static const Heat1DValidationCase MODAL_VALIDATION_CASES[] = {
    {0.01f,  1.00f,  0.00f},
    {0.10f,  1.00f,  0.00f}, /* original fixed Heat 1D baseline */
    {0.50f,  1.00f,  0.00f},
    {0.01f,  0.00f,  1.00f},
    {0.50f,  0.00f,  1.00f},
    {0.05f,  0.70f, -0.40f},
    {0.20f, -0.60f,  0.25f},
    {0.45f,  0.25f,  0.90f},
    {0.50f, -1.00f,  1.00f},
};
static const Heat1DValidationCase *active_validation_cases = VALIDATION_CASES;
static size_t active_validation_count = sizeof(VALIDATION_CASES) / sizeof(VALIDATION_CASES[0]);
static int active_coefficient_model = 0;
static int active_modal_model = 0;
static const char *active_training_mode = "modal";

static const float VALIDATION_REL_L2_MAX = 5e-2f;
static const float MODAL_VALIDATION_REL_L2_MAX = 1e-2f;
static float active_validation_rel_l2_max = 5e-2f;
typedef struct {
    MLP *mlp;
    Tensor **params;
    int n_params;
    Tensor *points;
    Tensor *zero_points;
    FILE *loss_file;
    int structured_mode;
} Heat1DLBFGSContext;

typedef struct {
    Tensor *prediction;
    Tensor *tau1;
    Tensor *tau2;
    Tensor *input1;
    Tensor *input2;
} Heat1DModalEvaluation;

static int heat1d_parameter_count(Tensor **params, int n_params);
static void heat1d_set_parameters(Tensor **params, int n_params, const lbfgsfloatval_t *x);
static void heat1d_copy_parameters(Tensor **params, int n_params, lbfgsfloatval_t *x);
static lbfgsfloatval_t heat1d_lbfgs_evaluate(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step);
static int heat1d_lbfgs_progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls);
static float evaluate_heat1d_case(MLP *mlp, Device device, float alpha, float a1, float a2);
static float heat1d_validation_max(MLP *mlp, Device device);
static Tensor *heat1d_model_inputs(Tensor *points);
static JetTensor *heat1d_model_jet_input(Tensor *points);
static Tensor *heat1d_training_residual(JetTensor *network, Tensor *points);
static Heat1DModalEvaluation heat1d_modal_evaluate(MLP *mlp, Tensor *points);
static void heat1d_modal_evaluation_free_inputs(Heat1DModalEvaluation *evaluation);

static int write_heat1d_metadata(const char *path, const float *lower, const float *upper, float max_validation_rel_l2);

int main(int argc, char **argv) {
    int n_steps = 2500;
    int n_col = 10000;
    int n_lbfgs_steps = 200;
    unsigned int seed = 1234;
    int surrogate_sizes[] = {5, 64, 64, 64, 1};
    int coefficient_sizes[] = {7, 128, 128, 128, 128, 1};
    int modal_sizes[] = {1, 32, 32, 32, 1};

    if(argc > 1) n_steps = atoi(argv[1]);
    if(argc > 2) n_col = atoi(argv[2]);
    if(argc > 3) seed = (unsigned int)strtoul(argv[3], NULL, 10);

    backend_init();
    Device device = backend_cuda_available() ? DEVICE_CUDA : DEVICE_CPU;
    if(argc > 4 && strcmp(argv[4], "cpu") == 0) device = DEVICE_CPU;
    if(argc > 4 && strcmp(argv[4], "cuda") == 0 && backend_cuda_available()) device = DEVICE_CUDA;
    if(argc > 5) n_lbfgs_steps = atoi(argv[5]);
    const char *training_mode = argc > 6 ? argv[6] : "modal";
    int modal = strcmp(training_mode, "modal") == 0;
    int fixed_baseline = strcmp(training_mode, "fixed") == 0;
    int mode1_only = strcmp(training_mode, "mode1") == 0;
    int mode2_only = strcmp(training_mode, "mode2") == 0;
    int structured_mode = modal || fixed_baseline || mode1_only || mode2_only;
    active_coefficient_model = mode1_only || mode2_only;
    active_modal_model = modal;
    active_training_mode = training_mode;
    if(!structured_mode && strcmp(training_mode, "surrogate") != 0){
        fprintf(stderr, "mode must be modal, surrogate, fixed, mode1, or mode2\n");
        return 1;
    }
    if(modal){
        active_validation_cases = MODAL_VALIDATION_CASES;
        active_validation_count = sizeof(MODAL_VALIDATION_CASES)
            / sizeof(MODAL_VALIDATION_CASES[0]);
        active_validation_rel_l2_max = MODAL_VALIDATION_REL_L2_MAX;
    } else if(mode1_only){
        active_validation_cases = MODE1_VALIDATION_CASES;
        active_validation_count = sizeof(MODE1_VALIDATION_CASES) / sizeof(MODE1_VALIDATION_CASES[0]);
    } else if(mode2_only){
        active_validation_cases = MODE2_VALIDATION_CASES;
        active_validation_count = sizeof(MODE2_VALIDATION_CASES) / sizeof(MODE2_VALIDATION_CASES[0]);
    } else if(fixed_baseline){
        active_validation_count = 1;
    }
    int n_zero_anchor = n_col / 5;
    if(n_zero_anchor < 256) n_zero_anchor = 256;
    if(n_steps <= 0 || n_col <= 0 || n_lbfgs_steps < 0){
        fprintf(stderr, "usage: %s [adam_steps] [collocation_points] [seed] [cpu|cuda] [lbfgs_steps] [modal|surrogate|fixed|mode1|mode2]\n", argv[0]);
        return 1;
    }
    srand(seed);

    int *sizes = active_modal_model ? modal_sizes
        : active_coefficient_model ? coefficient_sizes : surrogate_sizes;
    int n_sizes = active_coefficient_model ? 6 : 5;
    MLP *mlp = mlp_create(sizes, n_sizes, tanh_activation);
    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    if(device == DEVICE_CUDA){
        for(int i = 0; i < n_params; i++) tensor_to_cuda(params[i]);
    }
    Adam *adam = adam_create(params, n_params, 1e-3f);

    float lower[] = {0.0f, 0.0f, 0.01f, -1.0f, -1.0f};
    float upper[] = {1.0f, 1.0f, 0.50f, 1.0f, 1.0f};
    if(fixed_baseline){
        lower[HEAT1D_ALPHA] = upper[HEAT1D_ALPHA] = 0.10f;
        lower[HEAT1D_A1] = upper[HEAT1D_A1] = 1.00f;
        lower[HEAT1D_A2] = upper[HEAT1D_A2] = 0.00f;
    } else if(mode1_only){
        lower[HEAT1D_A1] = upper[HEAT1D_A1] = 1.00f;
        lower[HEAT1D_A2] = upper[HEAT1D_A2] = 0.00f;
    } else if(mode2_only){
        lower[HEAT1D_A1] = upper[HEAT1D_A1] = 0.00f;
        lower[HEAT1D_A2] = upper[HEAT1D_A2] = 1.00f;
    }

    float modal_lower[] = {0.0f};
    float modal_upper[] = {HEAT1D_MODAL_TAU_MAX};
    BoxDomain physical_domain = {.dim = HEAT1D_INPUT_DIM, .lower = lower, .upper = upper};
    BoxDomain modal_domain = {
        .dim = HEAT1D_MODAL_INPUT_DIM,
        .lower = modal_lower,
        .upper = modal_upper,
    };
    BoxDomain *domain = active_modal_model ? &modal_domain : &physical_domain;

    FILE *loss_file = fopen("projects/heat1d/files/heat1d_loss.csv", "w");
    if(!loss_file){
        printf("failed to open projects/heat1d/files/heat1d_loss.csv\n");
        return 1;
    }
    fprintf(loss_file, "phase,step,physics,total\n");

    struct timespec start;
    backend_sync(device);
    clock_gettime(CLOCK_MONOTONIC, &start);
    printf("Hyperparameters: adam_steps=%d lbfgs_steps=%d n_col=%d seed=%u device=%s mode=%s\n", n_steps, n_lbfgs_steps, n_col, seed, device == DEVICE_CUDA ? "CUDA" : "CPU", training_mode);
    for(int i = 0; i < n_steps; i++){ // Epochs
        Tape *tape = tape_create();
        set_curr_tape(tape);
        JetTape *jet_tape = jet_tape_create();
        set_curr_jet_tape(jet_tape);

        adam_zero_grad(adam);
        if(n_steps > 1){
            float progress = (float)i / (float)(n_steps - 1);
            adam->lr = 1e-5f + 0.5f * (1e-3f - 1e-5f)
                * (1.0f + cosf((float)M_PI * progress));
        }
        Tensor *points = sample_LHS_box(domain, n_col);
        if(device == DEVICE_CUDA) tensor_to_cuda(points);
        JetTensor *xj = heat1d_model_jet_input(points);
        JetTensor *N = jet_mlp_forward(mlp, xj);
        Tensor *residual = heat1d_training_residual(N, points);
        Tensor *physics_loss = residual_mse_loss(residual);
        Tensor *zero_points = NULL;
        JetTensor *zero_xj = NULL;
        Tensor *total_loss = physics_loss;
        if(!structured_mode){
            zero_points = sample_LHS_box(domain, n_zero_anchor);
            for(int row = 0; row < n_zero_anchor; row++){
                zero_points->data[row * HEAT1D_INPUT_DIM + HEAT1D_A1] = 0.0f;
                zero_points->data[row * HEAT1D_INPUT_DIM + HEAT1D_A2] = 0.0f;
            }
            if(device == DEVICE_CUDA) tensor_to_cuda(zero_points);
            zero_xj = heat1d_model_jet_input(zero_points);
            JetTensor *zero_N = jet_mlp_forward(mlp, zero_xj);
            Tensor *zero_residual = heat1d_parametric_ansatz_residual(zero_N, zero_points);
            Tensor *zero_loss = residual_mse_loss(zero_residual);
            total_loss = tensor_add(physics_loss, zero_loss);
        }

        backward(total_loss);
        adam_step(adam);

        if(i % 100 == 0 || i == n_steps - 1){
            tensor_to_cpu(physics_loss);
            printf("step=%d physics=%f total=%f\n", i, physics_loss->data[0], total_loss->data[0]);
            fprintf(loss_file, "adam,%d,%f,%f\n", i, physics_loss->data[0], total_loss->data[0]);
        }

        jet_tape_free_shallow(jet_tape);
        if(zero_xj) jet_free(zero_xj);
        if(zero_points) tensor_free(zero_points);
        jet_free(xj);
        tape_free(tape);
    }
    struct timespec end;
    backend_sync(device);
    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = elapsed_seconds(start, end);
    printf("training seconds=%f\n", elapsed);


    if(device == DEVICE_CUDA){
        backend_sync(device);
        for(int i = 0; i < n_params; i++) tensor_to_cpu(params[i]);
    }
    device = DEVICE_CPU;

    Tensor *lbfgs_points = sample_LHS_box(domain, n_col);
    Tensor *lbfgs_zero_points = NULL;
    if(!structured_mode){
        lbfgs_zero_points = sample_LHS_box(domain, n_zero_anchor);
        for(int row = 0; row < n_zero_anchor; row++){
            lbfgs_zero_points->data[row * HEAT1D_INPUT_DIM + HEAT1D_A1] = 0.0f;
            lbfgs_zero_points->data[row * HEAT1D_INPUT_DIM + HEAT1D_A2] = 0.0f;
        }
    }
    int n_flat_params = heat1d_parameter_count(params, n_params);
    lbfgsfloatval_t *lbfgs_x = lbfgs_malloc(n_flat_params);
    if(!lbfgs_x){
        fprintf(stderr, "failed to allocate L-BFGS parameter vector\n");
        tensor_free(lbfgs_points);
        if(lbfgs_zero_points) tensor_free(lbfgs_zero_points);
        adam_free(adam);
        free(params);
        mlp_free(mlp);
        return 1;
    }
    heat1d_copy_parameters(params, n_params, lbfgs_x);
    lbfgsfloatval_t *adam_x = lbfgs_malloc(n_flat_params);
    if(!adam_x){
        fprintf(stderr, "failed to allocate Adam checkpoint vector\n");
        lbfgs_free(lbfgs_x);
        tensor_free(lbfgs_points);
        if(lbfgs_zero_points) tensor_free(lbfgs_zero_points);
        adam_free(adam);
        free(params);
        mlp_free(mlp);
        return 1;
    }
    heat1d_copy_parameters(params, n_params, adam_x);
    float adam_validation_max = heat1d_validation_max(mlp, device);
    printf("Adam endpoint validation max rel_l2=%e\n", adam_validation_max);

    Heat1DLBFGSContext lbfgs_context = {
        .mlp = mlp,
        .params = params,
        .n_params = n_params,
        .points = lbfgs_points,
        .zero_points = lbfgs_zero_points,
        .loss_file = loss_file,
        .structured_mode = structured_mode,
    };
    lbfgs_parameter_t lbfgs_params;
    lbfgs_parameter_init(&lbfgs_params);
    lbfgs_params.max_iterations = n_lbfgs_steps;
    lbfgs_params.m = 10;

    if(n_lbfgs_steps > 0){
        lbfgsfloatval_t final_lbfgs_loss = 0.0;
        int lbfgs_status = lbfgs(
            n_flat_params,
            lbfgs_x,
            &final_lbfgs_loss,
            heat1d_lbfgs_evaluate,
            heat1d_lbfgs_progress,
            &lbfgs_context,
            &lbfgs_params
        );
        heat1d_set_parameters(params, n_params, lbfgs_x);
        printf("L-BFGS finished: status=%d (%s), physics=%f\n",
            lbfgs_status, lbfgs_strerror(lbfgs_status), (double)final_lbfgs_loss);
    }
    if(n_lbfgs_steps > 0){
        float lbfgs_validation_max = heat1d_validation_max(mlp, device);
        printf("L-BFGS endpoint validation max rel_l2=%e\n", lbfgs_validation_max);
        if(!isfinite(lbfgs_validation_max) || lbfgs_validation_max > adam_validation_max){
            fprintf(stderr, "L-BFGS worsened validation; restoring Adam checkpoint\n");
            heat1d_set_parameters(params, n_params, adam_x);
        }
    }
    lbfgs_free(adam_x);
    lbfgs_free(lbfgs_x);
    tensor_free(lbfgs_points);
    if(lbfgs_zero_points) tensor_free(lbfgs_zero_points);

    fclose(loss_file);
    FILE *validation_file = fopen(
        "projects/heat1d/files/heat1d_validation.csv", "w"
    );
    if(!validation_file){
        fprintf(stderr, "failed to open heat1d validation output\n");
        adam_free(adam);
        free(params);
        mlp_free(mlp);
        return 1;
    }
    fprintf(validation_file, "alpha,a1,a2,relative_l2\n");

    int validation_passed = 1;
    float max_validation_rel_l2 = 0.0f;
    size_t n_validation_cases = active_validation_count;
    for(size_t i = 0; i < n_validation_cases; i++){
        Heat1DValidationCase test_case = active_validation_cases[i];
        float rel_l2 = evaluate_heat1d_case(
            mlp, device, test_case.alpha, test_case.a1, test_case.a2
        );
        fprintf(validation_file, "%f,%f,%f,%.9e\n",
            test_case.alpha, test_case.a1, test_case.a2, rel_l2);
        printf("validation alpha=%f a1=%f a2=%f rel_l2=%e\n",
            test_case.alpha, test_case.a1, test_case.a2, rel_l2);

        if(!isfinite(rel_l2) || rel_l2 > active_validation_rel_l2_max){
            validation_passed = 0;
        }
        if(rel_l2 > max_validation_rel_l2){
            max_validation_rel_l2 = rel_l2;
        }
    }
    fclose(validation_file);

    if(!validation_passed){
        fprintf(stderr, "validation failed; model will not be exported\n");
        adam_free(adam);
        free(params);
        mlp_free(mlp);
        return 1;
    }

    if(device == DEVICE_CUDA){
        for(int i = 0; i < n_params; i++) tensor_to_cpu(params[i]);
    }
    const char *model_path = mode1_only ? "models/heat1d/v1/mode1.pinn"
        : mode2_only ? "models/heat1d/v1/mode2.pinn"
        : "models/heat1d/v1/model.pinn";
    const char *metadata_path = mode1_only ? "models/heat1d/v1/mode1.metadata.json"
        : mode2_only ? "models/heat1d/v1/mode2.metadata.json"
        : "models/heat1d/v1/metadata.json";
    float coefficient_lower[HEAT1D_COEFFICIENT_INPUT_DIM];
    float coefficient_upper[HEAT1D_COEFFICIENT_INPUT_DIM];
    for(int i = 0; i < HEAT1D_COEFFICIENT_INPUT_DIM; i++){
        coefficient_lower[i] = -1.0f;
        coefficient_upper[i] = 1.0f;
    }
    float modal_model_lower[] = {-1.0f};
    float modal_model_upper[] = {1.0f};
    const float *model_lower = active_modal_model ? modal_model_lower
        : active_coefficient_model ? coefficient_lower : lower;
    const float *model_upper = active_modal_model ? modal_model_upper
        : active_coefficient_model ? coefficient_upper : upper;
    int export_ok = pinn_model_save(
        model_path, mlp, model_lower, model_upper
    );
    if(device == DEVICE_CUDA){
        for(int i = 0; i < n_params; i++) tensor_to_cuda(params[i]);
    }
    if(!export_ok || !write_heat1d_metadata(
        metadata_path,
        lower,
        upper,
        max_validation_rel_l2
    )){
        fprintf(stderr, "failed to export Heat 1D model artifact\n");
        adam_free(adam);
        free(params);
        mlp_free(mlp);
        return 1;
    }

    int n_t_eval = 101;
    int n_x_eval = 101;
    int n_eval = n_t_eval * n_x_eval;
    const float eval_alpha = 0.1f;
    const float eval_a1 = mode2_only ? 0.0f : 1.0f;
    const float eval_a2 = mode2_only ? 1.0f : 0.0f;
    int eval_shape[2] = {n_eval, HEAT1D_INPUT_DIM};
    Tensor *eval_points = tensor_create(eval_shape, 2, 0);

    for(int i = 0; i < n_t_eval; i++){
        float t = (float)i / (float)(n_t_eval - 1);
        for(int j = 0; j < n_x_eval; j++){
            float x = (float)j / (float)(n_x_eval - 1);
            int idx = i * n_x_eval + j;
            eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_T] = t;
            eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_X] = x;
            eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_ALPHA] = eval_alpha;
            eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_A1] = eval_a1;
            eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_A2] = eval_a2;
        }
    }
    if(device == DEVICE_CUDA) tensor_to_cuda(eval_points);

    Tape *eval_tape = tape_create();
    set_curr_tape(eval_tape);
    struct timespec infer_start;
    backend_sync(device);
    clock_gettime(CLOCK_MONOTONIC, &infer_start);
    Tensor *eval_model_points = NULL;
    Heat1DModalEvaluation modal_evaluation = {0};
    Tensor *eval_pred;
    if(active_modal_model){
        modal_evaluation = heat1d_modal_evaluate(mlp, eval_points);
        eval_pred = modal_evaluation.prediction;
    } else {
        eval_model_points = heat1d_model_inputs(eval_points);
        Tensor *eval_raw = mlp_forward(mlp, eval_model_points);
        eval_pred = heat1d_parametric_ansatz(eval_raw, eval_points);
    }
    struct timespec infer_end;
    backend_sync(device);
    clock_gettime(CLOCK_MONOTONIC, &infer_end);
    double pinn_inference_seconds = elapsed_seconds(infer_start, infer_end);
    tensor_to_cpu(eval_points);
    tensor_to_cpu(eval_pred);

    FILE *pred_file = fopen("projects/heat1d/files/heat1d_predictions.csv", "w");
    if(!pred_file){
        printf("failed to open projects/heat1d/files/heat1d_predictions.csv\n");
        if(eval_model_points) tensor_free(eval_model_points);
        tape_free(eval_tape);
        heat1d_modal_evaluation_free_inputs(&modal_evaluation);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(pred_file, "t,x,u_pred,u_exact\n");

    for(int i = 0; i < n_t_eval; i++){
        for(int j = 0; j < n_x_eval; j++){
            int idx = i * n_x_eval + j;
            float t = eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_T];
            float x = eval_points->data[idx * HEAT1D_INPUT_DIM + HEAT1D_X];
            float u_pred = eval_pred->data[idx];
            float u_exact = heat1d_exact(t, x, eval_alpha, eval_a1, eval_a2);
            fprintf(pred_file, "%f,%f,%f,%f\n", t, x, u_pred, u_exact);
        }
    }

    fclose(pred_file);
    FILE *metrics_file = fopen("projects/heat1d/files/heat1d_metrics.csv", "w");
    if(!metrics_file){
        printf("failed to open projects/heat1d/files/heat1d_metrics.csv\n");
        if(eval_model_points) tensor_free(eval_model_points);
        tape_free(eval_tape);
        heat1d_modal_evaluation_free_inputs(&modal_evaluation);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(metrics_file, "metric,value\n");
    fprintf(metrics_file, "training_seconds,%f\n", elapsed);
    fprintf(metrics_file, "pinn_inference_seconds,%f\n", pinn_inference_seconds);
    fprintf(metrics_file, "eval_points,%d\n", n_eval);
    fclose(metrics_file);
    printf("wrote projects/heat1d/files/heat1d_loss.csv\n");
    printf("wrote projects/heat1d/files/heat1d_predictions.csv\n");
    printf("wrote projects/heat1d/files/heat1d_metrics.csv\n");
    if(eval_model_points) tensor_free(eval_model_points);
    tape_free(eval_tape);
    heat1d_modal_evaluation_free_inputs(&modal_evaluation);
    tensor_free(eval_points);
    adam_free(adam);
    free(params);
    mlp_free(mlp);
    return 0;
}
static float evaluate_heat1d_case(MLP *mlp, Device device, float alpha, float a1, float a2){
    const int n_t_eval = 101;
    const int n_x_eval = 101;
    const int n_eval = n_t_eval * n_x_eval;
    int shape[2] = {n_eval, HEAT1D_INPUT_DIM};
    Tensor *points = tensor_create(shape, 2, 0);

    for(int i = 0; i < n_t_eval; i++){
        float t = (float)i / (float)(n_t_eval - 1);
        for(int j = 0; j < n_x_eval; j++){
            float x = (float)j / (float)(n_x_eval - 1);
            int index = i * n_x_eval + j;
            points->data[index * HEAT1D_INPUT_DIM + HEAT1D_T] = t;
            points->data[index * HEAT1D_INPUT_DIM + HEAT1D_X] = x;
            points->data[index * HEAT1D_INPUT_DIM + HEAT1D_ALPHA] = alpha;
            points->data[index * HEAT1D_INPUT_DIM + HEAT1D_A1] = a1;
            points->data[index * HEAT1D_INPUT_DIM + HEAT1D_A2] = a2;
        }
    }

    if(device == DEVICE_CUDA) tensor_to_cuda(points);
    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *model_points = NULL;
    Heat1DModalEvaluation modal_evaluation = {0};
    Tensor *prediction;
    if(active_modal_model){
        modal_evaluation = heat1d_modal_evaluate(mlp, points);
        prediction = modal_evaluation.prediction;
    } else {
        model_points = heat1d_model_inputs(points);
        Tensor *raw = mlp_forward(mlp, model_points);
        prediction = heat1d_parametric_ansatz(raw, points);
    }
    tensor_to_cpu(prediction);

    double squared_error = 0.0;
    double squared_reference = 0.0;
    for(int i = 0; i < n_t_eval; i++){
        float t = (float)i / (float)(n_t_eval - 1);
        for(int j = 0; j < n_x_eval; j++){
            float x = (float)j / (float)(n_x_eval - 1);
            int index = i * n_x_eval + j;
            float exact = heat1d_exact(t, x, alpha, a1, a2);
            double error = (double)prediction->data[index] - exact;
            squared_error += error * error;
            squared_reference += (double)exact * exact;
        }
    }

    set_curr_tape(NULL);
    tape_free(tape);
    if(model_points) tensor_free(model_points);
    heat1d_modal_evaluation_free_inputs(&modal_evaluation);
    tensor_free(points);
    return squared_reference > 0.0
        ? (float)sqrt(squared_error / squared_reference)
        : INFINITY;
}

static Tensor *heat1d_model_inputs(Tensor *points){
    return active_coefficient_model
        ? heat1d_coefficient_features(points)
        : heat1d_normalize_inputs(points);
}

static JetTensor *heat1d_model_jet_input(Tensor *points){
    return active_modal_model
        ? heat1d_modal_jet_input(points)
        : active_coefficient_model
        ? heat1d_coefficient_feature_jet_input(points)
        : heat1d_normalized_jet_input(points);
}

static Tensor *heat1d_training_residual(JetTensor *network, Tensor *points){
    return active_modal_model
        ? heat1d_modal_residual(network, points)
        : heat1d_parametric_ansatz_residual(network, points);
}

static Heat1DModalEvaluation heat1d_modal_evaluate(MLP *mlp, Tensor *points){
    Heat1DModalEvaluation evaluation = {0};
    evaluation.tau1 = heat1d_modal_tau_points(points, 1);
    evaluation.tau2 = heat1d_modal_tau_points(points, 2);
    if(!evaluation.tau1 || !evaluation.tau2) return evaluation;

    evaluation.input1 = heat1d_modal_normalize_tau(evaluation.tau1);
    evaluation.input2 = heat1d_modal_normalize_tau(evaluation.tau2);
    if(!evaluation.input1 || !evaluation.input2) return evaluation;

    Tensor *network1 = mlp_forward(mlp, evaluation.input1);
    Tensor *network2 = mlp_forward(mlp, evaluation.input2);
    Tensor *q1 = heat1d_modal_coefficient(network1, evaluation.tau1);
    Tensor *q2 = heat1d_modal_coefficient(network2, evaluation.tau2);
    evaluation.prediction = heat1d_modal_reconstruct(q1, q2, points);
    return evaluation;
}

static void heat1d_modal_evaluation_free_inputs(Heat1DModalEvaluation *evaluation){
    if(!evaluation) return;
    tensor_free(evaluation->input1);
    tensor_free(evaluation->input2);
    tensor_free(evaluation->tau1);
    tensor_free(evaluation->tau2);
    evaluation->input1 = NULL;
    evaluation->input2 = NULL;
    evaluation->tau1 = NULL;
    evaluation->tau2 = NULL;
    evaluation->prediction = NULL;
}

static float heat1d_validation_max(MLP *mlp, Device device){
    float max_rel_l2 = 0.0f;
    size_t count = active_validation_count;
    for(size_t i = 0; i < count; i++){
        Heat1DValidationCase test_case = active_validation_cases[i];
        float rel_l2 = evaluate_heat1d_case(mlp, device, test_case.alpha, test_case.a1, test_case.a2);
        if(!isfinite(rel_l2)) return INFINITY;
        if(rel_l2 > max_rel_l2) max_rel_l2 = rel_l2;
    }
    return max_rel_l2;
}

static int write_heat1d_metadata(
    const char *path,
    const float *lower,
    const float *upper,
    float max_validation_rel_l2
){
    FILE *file = fopen(path, "w");
    if(!file) return 0;

    const char *architecture = active_modal_model
        ? "[1, 32, 32, 32, 1]"
        : active_coefficient_model
        ? "[7, 128, 128, 128, 128, 1]"
        : "[5, 64, 64, 64, 1]";
    const char *representation = active_modal_model
        ? "dimensionless_modal_tau_v1"
        : active_coefficient_model
        ? "normalized_fourier_v1"
        : "normalized_raw_v1";
    const char *input_order = active_modal_model
        ? "[\"tau_n\"]"
        : active_coefficient_model
        ? "[\"t\", \"x\", \"alpha\"]"
        : "[\"t\", \"x\", \"alpha\", \"a1\", \"a2\"]";
    int ok;
    if(active_modal_model){
        ok = fprintf(file,
            "{\n"
            "  \"equation\": \"heat1d-modal\",\n"
            "  \"format_version\": 1,\n"
            "  \"architecture\": %s,\n"
            "  \"activation\": \"tanh_hidden_linear_output\",\n"
            "  \"input_representation\": \"%s\",\n"
            "  \"model_input_order\": %s,\n"
            "  \"physical_input_order\": [\"t\", \"x\", \"alpha\", \"a1\", \"a2\"],\n"
            "  \"physical_input_lower\": [%g, %g, %g, %g, %g],\n"
            "  \"physical_input_upper\": [%g, %g, %g, %g, %g],\n"
            "  \"tau_range\": [0, %.9g],\n"
            "  \"coefficient_ansatz\": \"q(tau) = (1 + tau * N(tau)) / (1 + tau)\",\n"
            "  \"reconstruction\": \"a1*sin(pi*x)*q(alpha*pi^2*t) + a2*sin(2*pi*x)*q(4*alpha*pi^2*t)\",\n"
            "  \"validation_rel_l2_max\": %.9g,\n"
            "  \"validation_rel_l2_threshold\": %.9g\n"
            "}\n",
            architecture, representation, input_order,
            lower[0], lower[1], lower[2], lower[3], lower[4],
            upper[0], upper[1], upper[2], upper[3], upper[4],
            HEAT1D_MODAL_TAU_MAX,
            max_validation_rel_l2, active_validation_rel_l2_max
        ) >= 0;
    } else if(active_coefficient_model){
        ok = fprintf(file,
            "{\n"
            "  \"equation\": \"heat1d-%s\",\n"
            "  \"format_version\": 1,\n"
            "  \"architecture\": %s,\n"
            "  \"activation\": \"tanh_hidden_linear_output\",\n"
            "  \"input_representation\": \"%s\",\n"
            "  \"input_order\": %s,\n"
            "  \"feature_order\": [\"t_n\", \"x_n\", \"alpha_n\", \"sin_pi_x\", \"cos_pi_x\", \"sin_2pi_x\", \"cos_2pi_x\"],\n"
            "  \"input_lower\": [%g, %g, %g],\n"
            "  \"input_upper\": [%g, %g, %g],\n"
            "  \"basis_amplitudes\": [%g, %g],\n"
            "  \"validation_rel_l2_max\": %.9g,\n"
            "  \"validation_rel_l2_threshold\": %.9g\n"
            "}\n",
            active_training_mode, architecture, representation, input_order,
            lower[0], lower[1], lower[2],
            upper[0], upper[1], upper[2],
            lower[3], lower[4],
            max_validation_rel_l2, active_validation_rel_l2_max
        ) >= 0;
    } else {
        ok = fprintf(file,
            "{\n"
            "  \"equation\": \"heat1d-%s\",\n"
            "  \"format_version\": 1,\n"
            "  \"architecture\": %s,\n"
            "  \"activation\": \"tanh_hidden_linear_output\",\n"
            "  \"input_representation\": \"%s\",\n"
            "  \"input_order\": %s,\n"
            "  \"input_lower\": [%g, %g, %g, %g, %g],\n"
            "  \"input_upper\": [%g, %g, %g, %g, %g],\n"
            "  \"validation_rel_l2_max\": %.9g,\n"
            "  \"validation_rel_l2_threshold\": %.9g\n"
            "}\n",
            active_training_mode, architecture, representation, input_order,
            lower[0], lower[1], lower[2], lower[3], lower[4],
            upper[0], upper[1], upper[2], upper[3], upper[4],
            max_validation_rel_l2, active_validation_rel_l2_max
        ) >= 0;
    }

    fclose(file);
    return ok;
}
static int heat1d_parameter_count(Tensor **params, int n_params){
    int count = 0;
    for(int i = 0; i < n_params; i++) count += params[i]->size;
    return count;
}

static void heat1d_set_parameters(Tensor **params, int n_params, const lbfgsfloatval_t *x){
    int offset = 0;
    for(int i = 0; i < n_params; i++){
        for(int j = 0; j < params[i]->size; j++) params[i]->data[j] = (float)x[offset++];
    }
}

static void heat1d_copy_parameters(Tensor **params, int n_params, lbfgsfloatval_t *x){
    int offset = 0;
    for(int i = 0; i < n_params; i++){
        for(int j = 0; j < params[i]->size; j++) x[offset++] = params[i]->data[j];
    }
}

static lbfgsfloatval_t heat1d_lbfgs_evaluate(void *instance, const lbfgsfloatval_t *x, lbfgsfloatval_t *g, const int n, const lbfgsfloatval_t step){
    (void)n;
    (void)step;
    Heat1DLBFGSContext *context = instance;
    heat1d_set_parameters(context->params, context->n_params, x);
    Tape *tape = tape_create();
    set_curr_tape(tape);
    JetTape *jet_tape = jet_tape_create();
    set_curr_jet_tape(jet_tape);
    sgd_zero_grad(context->params, context->n_params);
    JetTensor *input = heat1d_model_jet_input(context->points);
    JetTensor *network = jet_mlp_forward(context->mlp, input);
    Tensor *residual = heat1d_training_residual(network, context->points);
    Tensor *loss = residual_mse_loss(residual);
    JetTensor *zero_input = NULL;
    if(!context->structured_mode){
        zero_input = heat1d_model_jet_input(context->zero_points);
        JetTensor *zero_network = jet_mlp_forward(context->mlp, zero_input);
        Tensor *zero_residual = heat1d_parametric_ansatz_residual(
            zero_network, context->zero_points
        );
        loss = tensor_add(loss, residual_mse_loss(zero_residual));
    }
    lbfgsfloatval_t loss_value = loss->data[0];
    backward(loss);
    int offset = 0;
    for(int i = 0; i < context->n_params; i++){
        for(int j = 0; j < context->params[i]->size; j++) g[offset++] = context->params[i]->grad[j];
    }
    jet_tape_free_shallow(jet_tape);
    if(zero_input) jet_free(zero_input);
    jet_free(input);
    set_curr_tape(NULL);
    tape_free(tape);
    return loss_value;
}

static int heat1d_lbfgs_progress(void *instance, const lbfgsfloatval_t *x, const lbfgsfloatval_t *g, const lbfgsfloatval_t fx, const lbfgsfloatval_t xnorm, const lbfgsfloatval_t gnorm, const lbfgsfloatval_t step, int n, int k, int ls){
    (void)x;
    (void)g;
    (void)xnorm;
    (void)gnorm;
    (void)step;
    (void)n;
    (void)ls;
    Heat1DLBFGSContext *context = instance;
    if(k == 1 || k % 10 == 0){
        printf("lbfgs step=%d physics=%f\n", k, (double)fx);
        fprintf(context->loss_file, "lbfgs,%d,%f,%f\n", k, (double)fx, (double)fx);
    }
    return 0;
}
