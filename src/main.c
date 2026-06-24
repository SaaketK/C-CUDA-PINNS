/*
 * src/main.c
 *
 * Placeholder executable for the framework. 
 * Smoke tests live under src/tests.
 * cd /Users/saaketk/PycharmProjects/PINN && cmake --build C-CUDA-PINNs/build --target pinn_main && (cd C-CUDA-PINNs && ./build/pinn_main) && MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig .venv/bin/python C-CUDA-PINNs/examples/heat1d/plot_heat1d.py
*/

#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/nn/mlp.h"
#include "pinn/nn/optimizer.h"
#include "pinn/pinn/residual.h"
#include "pinn/pinn/sampler.h"
#include "pinn/pinn/trainer.h"
#include "pinn/autodiff/jet.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

static float heat1d_exact(float t, float x, float alpha){
    return expf(-alpha * (float)M_PI * (float)M_PI * t) * sinf((float)M_PI * x);
}

int main(void) {
    int n_steps = 2500;
    int n_col = 200;
    int sizes[] = {2, 64, 64, 64, 1};

    MLP *mlp = mlp_create(sizes, 5);

    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Adam *adam = adam_create(params, n_params, 1e-3f);

    float lower[] = {0.0f, 0.0f};
    float upper[] = {1.0f, 1.0f};

    BoxDomain domain = {.dim = 2, .lower = lower, .upper = upper};
    Heat1DParams heat = {.alpha = 0.1f};

    FILE *loss_file = fopen("examples/heat1d/heat1d_loss.csv", "w");
    if(!loss_file){
        printf("failed to open examples/heat1d/heat1d_loss.csv\n");
        return 1;
    }
    fprintf(loss_file, "step,physics,total\n");

    clock_t start = clock();
    printf("Hyperparameters: n_steps=%d n_col=%d\n", n_steps, n_col);
    for(int i = 0; i < n_steps; i++){ // Epochs
        Tape *tape = tape_create();
        set_curr_tape(tape);
        JetTape *jet_tape = jet_tape_create();
        set_curr_jet_tape(jet_tape);

        adam_zero_grad(adam);
        Tensor *points = sample_LHS_box(&domain, n_col);
        JetTensor *xj = jet_create_input(points, 2);
        JetTensor *N = jet_mlp_forward(mlp, xj);
        Tensor *residual = heat1d_ansatz_residual(N, points, &heat);
        Tensor *loss = residual_mse_loss(residual);
        Tensor *total_loss = loss;

        backward(total_loss);
        adam_step(adam);

        if(i % 100 == 0 || i == n_steps - 1){
            printf("step=%d physics=%f total=%f\n", i, loss->data[0], total_loss->data[0]);
            fprintf(loss_file, "%d,%f,%f\n", i, loss->data[0], total_loss->data[0]);
        }

        jet_tape_free_shallow(jet_tape);
        jet_free(xj);
        tape_free(tape);
    }
    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("training seconds=%f\n", elapsed);
    fclose(loss_file);

    int n_t_eval = 101;
    int n_x_eval = 101;
    int n_eval = n_t_eval * n_x_eval;
    int eval_shape[2] = {n_eval, 2};
    Tensor *eval_points = tensor_create(eval_shape, 2, 0);

    for(int i = 0; i < n_t_eval; i++){
        float t = (float)i / (float)(n_t_eval - 1);
        for(int j = 0; j < n_x_eval; j++){
            float x = (float)j / (float)(n_x_eval - 1);
            int idx = i * n_x_eval + j;
            eval_points->data[idx * 2] = t;
            eval_points->data[idx * 2 + 1] = x;
        }
    }

    Tape *eval_tape = tape_create();
    set_curr_tape(eval_tape);
    clock_t infer_start = clock();
    Tensor *eval_raw = mlp_forward(mlp, eval_points);
    Tensor *eval_pred = heat1d_ansatz(eval_raw, eval_points, &heat);
    double pinn_inference_seconds = (double)(clock() - infer_start) / CLOCKS_PER_SEC;

    FILE *pred_file = fopen("examples/heat1d/heat1d_predictions.csv", "w");
    if(!pred_file){
        printf("failed to open examples/heat1d/heat1d_predictions.csv\n");
        tape_free(eval_tape);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(pred_file, "t,x,u_pred,u_exact\n");

    for(int i = 0; i < n_t_eval; i++){
        for(int j = 0; j < n_x_eval; j++){
            int idx = i * n_x_eval + j;
            float t = eval_points->data[idx * 2];
            float x = eval_points->data[idx * 2 + 1];
            float u_pred = eval_pred->data[idx];
            float u_exact = heat1d_exact(t, x, heat.alpha);
            fprintf(pred_file, "%f,%f,%f,%f\n", t, x, u_pred, u_exact);
        }
    }

    fclose(pred_file);
    FILE *metrics_file = fopen("examples/heat1d/heat1d_metrics.csv", "w");
    if(!metrics_file){
        printf("failed to open examples/heat1d/heat1d_metrics.csv\n");
        tape_free(eval_tape);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(metrics_file, "metric,value\n");
    fprintf(metrics_file, "training_seconds,%f\n", elapsed);
    fprintf(metrics_file, "pinn_inference_seconds,%f\n", pinn_inference_seconds);
    fprintf(metrics_file, "eval_points,%d\n", n_eval);
    fclose(metrics_file);
    printf("wrote examples/heat1d/heat1d_loss.csv\n");
    printf("wrote examples/heat1d/heat1d_predictions.csv\n");
    printf("wrote examples/heat1d/heat1d_metrics.csv\n");
    tape_free(eval_tape);
    tensor_free(eval_points);
    adam_free(adam);
    free(params);
    mlp_free(mlp);
    return 0;
}
