#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/nn/mlp.h"
#include "pinn/nn/optimizer.h"
#include "pinn/nn/lbfgs.h"
#include "pinn/pinn/residual.h"
#include "pinn/pinn/sampler.h"
#include "pinn/pinn/trainer.h"
#include "pinn/autodiff/jet.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950
#endif

static float heat2d_exact(float t, float x, float y, float alpha_x, float alpha_y){
    float first_mode = expf(-(alpha_x + alpha_y) * (float)M_PI * (float)M_PI * t)
        * sinf((float)M_PI * x) * sinf((float)M_PI * y);
    float x_harmonic = 0.5f * expf(-(4.0f * alpha_x + alpha_y) * (float)M_PI * (float)M_PI * t)
        * sinf(2.0f * (float)M_PI * x) * sinf((float)M_PI * y);
    return first_mode + x_harmonic;
}

typedef struct {
    MLP *mlp;
    Tensor **params;
    int n_params;
    Tensor *points;
    Heat2DParams *heat;
    FILE *loss_file;
} Heat2DLBFGSContext;

static int parameter_count(Tensor **params, int n_params){
    int count = 0;
    for(int i = 0; i < n_params; i++) count += params[i]->size;
    return count;
}

static void set_parameters_from_lbfgs(
    Tensor **params, int n_params, const lbfgsfloatval_t *x
){
    int offset = 0;
    for(int i = 0; i < n_params; i++){
        for(int j = 0; j < params[i]->size; j++){
            params[i]->data[j] = (float)x[offset++];
        }
    }
}

static void copy_parameters_to_lbfgs(
    Tensor **params, int n_params, lbfgsfloatval_t *x
){
    int offset = 0;
    for(int i = 0; i < n_params; i++){
        for(int j = 0; j < params[i]->size; j++){
            x[offset++] = params[i]->data[j];
        }
    }
}

static lbfgsfloatval_t heat2d_lbfgs_evaluate(
    void *instance,
    const lbfgsfloatval_t *x,
    lbfgsfloatval_t *g,
    const int n,
    const lbfgsfloatval_t step
){
    (void)n;
    (void)step;
    Heat2DLBFGSContext *context = instance;
    set_parameters_from_lbfgs(context->params, context->n_params, x);
    Tape *tape = tape_create();
    set_curr_tape(tape);
    JetTape *jet_tape = jet_tape_create();
    set_curr_jet_tape(jet_tape);
    sgd_zero_grad(context->params, context->n_params);
    JetTensor *xj = jet_create_input(context->points, 3);
    JetTensor *N = jet_mlp_forward(context->mlp, xj);
    Tensor *residual = heat2d_ansatz_residual(N, context->points, context->heat);
    Tensor *loss = residual_mse_loss(residual);
    lbfgsfloatval_t loss_value = loss->data[0];
    backward(loss);
    int offset = 0;
    for(int i = 0; i < context->n_params; i++){
        for(int j = 0; j < context->params[i]->size; j++){
            g[offset++] = context->params[i]->grad[j];
        }
    }
    jet_tape_free_shallow(jet_tape);
    tensor_free(xj->d1);
    tensor_free(xj->d2);
    jet_free_shallow(xj);
    tape_free(tape);
    return loss_value;
}

static int heat2d_lbfgs_progress(
    void *instance,
    const lbfgsfloatval_t *x,
    const lbfgsfloatval_t *g,
    const lbfgsfloatval_t fx,
    const lbfgsfloatval_t xnorm,
    const lbfgsfloatval_t gnorm,
    const lbfgsfloatval_t step,
    int n,
    int k,
    int ls
){
    (void)x;
    (void)g;
    (void)xnorm;
    (void)gnorm;
    (void)step;
    (void)n;
    (void)ls;
    Heat2DLBFGSContext *context = instance;
    if(k == 1 || k % 10 == 0){
        printf("lbfgs step=%d physics=%f\n", k, (double)fx);
        fprintf(context->loss_file, "lbfgs,%d,%f,%f\n", k, (double)fx, (double)fx);
    }
    return 0;
}

int main(int argc, char **argv) {
    int n_adam_steps = 2500;
    int n_col = 500;
    int n_lbfgs_steps = 200;
    if(argc > 1){
        n_adam_steps = atoi(argv[1]);
    }
    if(argc > 2){
        n_col = atoi(argv[2]);
    }
    if(argc > 3){
        n_lbfgs_steps = atoi(argv[3]);
    }
    if(n_adam_steps <= 0 || n_col <= 0 || n_lbfgs_steps < 0){
        fprintf(stderr, "usage: %s [positive_adam_steps] [positive_n_collocation] [nonnegative_lbfgs_steps]\n", argv[0]);
        return 1;
    }
    int sizes[] = {3, 64, 64, 64, 1};

    MLP *mlp = mlp_create(sizes, 5, tanh_activation);

    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Adam *adam = adam_create(params, n_params, 1e-3f);

    float lower[] = {0.0f, 0.0f, 0.0f};
    float upper[] = {1.0f, 1.0f, 1.0f};

    BoxDomain domain = {.dim = 3, .lower = lower, .upper = upper};
    Heat2DParams heat = {.alpha_x = 0.1f, .alpha_y = 0.02f};

    FILE *loss_file = fopen("examples/heat2d/files/heat2d_loss.csv", "w");
    if(!loss_file){
        printf("failed to open examples/heat2d/files/heat2d_loss.csv\n");
        return 1;
    }
    fprintf(loss_file, "phase,step,physics,total\n");

    clock_t start = clock();
    printf("Hyperparameters: adam_steps=%d lbfgs_steps=%d n_col=%d\n",
        n_adam_steps, n_lbfgs_steps, n_col);
    for(int i = 0; i < n_adam_steps; i++){
        Tape *tape = tape_create();
        set_curr_tape(tape);
        JetTape *jet_tape = jet_tape_create();
        set_curr_jet_tape(jet_tape);

        adam_zero_grad(adam);
        Tensor *points = sample_LHS_box(&domain, n_col);
        JetTensor *xj = jet_create_input(points, 3);
        JetTensor *N = jet_mlp_forward(mlp, xj);
        Tensor *residual = heat2d_ansatz_residual(N, points, &heat);
        Tensor *loss = residual_mse_loss(residual);
        Tensor *total_loss = loss;

        backward(total_loss);
        adam_step(adam);

        if(i % 100 == 0 || i == n_adam_steps - 1){
            printf("adam step=%d physics=%f total=%f\n", i, loss->data[0], total_loss->data[0]);
            fprintf(loss_file, "adam,%d,%f,%f\n", i, loss->data[0], total_loss->data[0]);
        }

        jet_tape_free_shallow(jet_tape);
        jet_free(xj);
        tape_free(tape);
    }
    adam_free(adam);

    Tensor *lbfgs_points = sample_LHS_box(&domain, n_col);
    int n_flat_params = parameter_count(params, n_params);
    lbfgsfloatval_t *lbfgs_x = lbfgs_malloc(n_flat_params);
    if(!lbfgs_x){
        printf("failed to allocate L-BFGS parameter vector\n");
        tensor_free(lbfgs_points);
        fclose(loss_file);
        free(params);
        mlp_free(mlp);
        return 1;
    }
    copy_parameters_to_lbfgs(params, n_params, lbfgs_x);

    Heat2DLBFGSContext lbfgs_context = {
        .mlp = mlp,
        .params = params,
        .n_params = n_params,
        .points = lbfgs_points,
        .heat = &heat,
        .loss_file = loss_file,
    };
    lbfgs_parameter_t lbfgs_params;
    lbfgs_parameter_init(&lbfgs_params);
    lbfgs_params.max_iterations = n_lbfgs_steps;
    lbfgs_params.m = 10;

    lbfgsfloatval_t final_lbfgs_loss = 0.0;
    int lbfgs_status = LBFGS_SUCCESS;
    if(n_lbfgs_steps > 0){
        lbfgs_status = lbfgs(
            n_flat_params,
            lbfgs_x,
            &final_lbfgs_loss,
            heat2d_lbfgs_evaluate,
            heat2d_lbfgs_progress,
            &lbfgs_context,
            &lbfgs_params
        );
        set_parameters_from_lbfgs(params, n_params, lbfgs_x);
        printf("L-BFGS finished: status=%d (%s), physics=%f\n",
            lbfgs_status, lbfgs_strerror(lbfgs_status), (double)final_lbfgs_loss);
    }
    lbfgs_free(lbfgs_x);
    tensor_free(lbfgs_points);

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("training seconds=%f\n", elapsed);
    fclose(loss_file);

    int n_t_eval = 25;
    int n_x_eval = 25;
    int n_y_eval = 25;
    int n_eval = n_t_eval * n_x_eval * n_y_eval;
    int eval_shape[2] = {n_eval, 3};
    Tensor *eval_points = tensor_create(eval_shape, 2, 0);

    for(int i = 0; i < n_t_eval; i++){
        float t = (float)i / (float)(n_t_eval - 1);
        for(int j = 0; j < n_x_eval; j++){
            float x = (float)j / (float)(n_x_eval - 1);
            for(int k = 0; k < n_y_eval; k++){
                float y = (float)k / (float)(n_y_eval - 1);
                int idx = (i * n_x_eval + j) * n_y_eval + k;
                eval_points->data[idx * 3] = t;
                eval_points->data[idx * 3 + 1] = x;
                eval_points->data[idx * 3 + 2] = y;
            }
        }
    }

    Tape *eval_tape = tape_create();
    set_curr_tape(eval_tape);
    clock_t infer_start = clock();
    Tensor *eval_raw = mlp_forward(mlp, eval_points);
    Tensor *eval_pred = heat2d_ansatz(eval_raw, eval_points, &heat);
    double pinn_inference_seconds = (double)(clock() - infer_start) / CLOCKS_PER_SEC;

    FILE *pred_file = fopen("examples/heat2d/files/heat2d_predictions.csv", "w");
    if(!pred_file){
        printf("failed to open examples/heat2d/files/heat2d_predictions.csv\n");
        tape_free(eval_tape);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(pred_file, "t,x,y,u_pred,u_exact\n");

    double abs_error_sum = 0.0;
    double squared_error_sum = 0.0;
    float max_abs_error = 0.0f;
    for(int i = 0; i < n_t_eval; i++){
        for(int j = 0; j < n_x_eval; j++){
            for(int k = 0; k < n_y_eval; k++){
                int idx = (i * n_x_eval + j) * n_y_eval + k;
                float t = eval_points->data[idx * 3];
                float x = eval_points->data[idx * 3 + 1];
                float y = eval_points->data[idx * 3 + 2];
                float u_pred = eval_pred->data[idx];
                float u_exact = heat2d_exact(t, x, y, heat.alpha_x, heat.alpha_y);
                float abs_error = fabsf(u_pred - u_exact);
                abs_error_sum += abs_error;
                squared_error_sum += (double)abs_error * abs_error;
                if(abs_error > max_abs_error){
                    max_abs_error = abs_error;
                }
                fprintf(pred_file, "%f,%f,%f,%f,%f\n", t, x, y, u_pred, u_exact);
            }
        }
    }

    fclose(pred_file);
    FILE *metrics_file = fopen("examples/heat2d/files/heat2d_metrics.csv", "w");
    if(!metrics_file){
        printf("failed to open examples/heat2d/files/heat2d_metrics.csv\n");
        tape_free(eval_tape);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(metrics_file, "metric,value\n");
    fprintf(metrics_file, "training_seconds,%f\n", elapsed);
    fprintf(metrics_file, "adam_steps,%d\n", n_adam_steps);
    fprintf(metrics_file, "lbfgs_steps,%d\n", n_lbfgs_steps);
    fprintf(metrics_file, "lbfgs_status,%d\n", lbfgs_status);
    fprintf(metrics_file, "lbfgs_final_physics,%f\n", (double)final_lbfgs_loss);
    fprintf(metrics_file, "collocation_points,%d\n", n_col);
    fprintf(metrics_file, "pinn_inference_seconds,%f\n", pinn_inference_seconds);
    fprintf(metrics_file, "eval_points,%d\n", n_eval);
    fprintf(metrics_file, "mae,%f\n", abs_error_sum / n_eval);
    fprintf(metrics_file, "rmse,%f\n", sqrt(squared_error_sum / n_eval));
    fprintf(metrics_file, "max_abs_error,%f\n", max_abs_error);
    fclose(metrics_file);
    printf("wrote examples/heat2d/files/heat2d_loss.csv\n");
    printf("wrote examples/heat2d/files/heat2d_predictions.csv\n");
    printf("wrote examples/heat2d/files/heat2d_metrics.csv\n");
    tape_free(eval_tape);
    tensor_free(eval_points);
    free(params);
    mlp_free(mlp);
    return 0;
}
