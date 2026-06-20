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
    int n_bc = 100;
    int n_ic = 100;
    int sizes[] = {2, 64, 64, 64, 1};

    MLP *mlp = mlp_create(sizes, 5);

    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Adam *adam = adam_create(params, n_params, 1e-3f);

    float lower[] = {0.0f, 0.0f};
    float upper[] = {1.0f, 1.0f};

    BoxDomain domain = {.dim = 2, .lower = lower, .upper = upper};
    Heat1DParams heat = {.alpha = 0.1f};

    int ic_shape[] = {n_ic, 1};
    Tensor *ic_points = sample_fixed_dim_box(&domain, n_ic, 0, 0.0f);
    Tensor *ic_targets = tensor_create(ic_shape, 2, 0);

    
    int bc_shape[] = {n_bc, 1};
    Tensor *bc_left = sample_fixed_dim_box(&domain, n_bc, 1, 0.0f);
    Tensor *bc_right = sample_fixed_dim_box(&domain, n_bc, 1, 1.0f);
    Tensor *bc_zero_left = tensor_create(bc_shape, 2, 0);
    Tensor *bc_zero_right = tensor_create(bc_shape, 2, 0);

    for(int i = 0; i < n_ic; i++){
        float x = ic_points->data[i * 2 + 1];
        ic_targets->data[i] = sinf(M_PI * x);
    }

    FILE *loss_file = fopen("examples/heat1d/files/heat1d_loss.csv", "w");
    if(!loss_file){
        printf("failed to open examples/heat1d/files/heat1d_loss.csv\n");
        return 1;
    }
    fprintf(loss_file, "step,physics,ic,bc,total\n");

    clock_t start = clock();
    printf("Hyperparameters: n_steps=%d n_col=%d n_bc=%d n_ic=%d\n", n_steps, n_col, n_bc, n_ic);
    for(int i = 0; i < n_steps; i++){ // Epochs
        Tape *tape = tape_create();
        set_curr_tape(tape);
        JetTape *jet_tape = jet_tape_create();
        set_curr_jet_tape(jet_tape);

        adam_zero_grad(adam);
        Tensor *points = sample_uniform_box(&domain, n_col);
        JetTensor *xj = jet_create_input(points, 2);
        JetTensor *u = jet_mlp_forward(mlp, xj);
        Tensor *residual = heat1d_residual(u, points, &heat);
        Tensor *loss = residual_mse_loss(residual);
        Tensor *ic_pred = mlp_forward(mlp, ic_points);
        Tensor *ic_loss = tensor_mse(ic_pred, ic_targets);
        Tensor *bc_left_pred = mlp_forward(mlp, bc_left);
        Tensor *bc_right_pred = mlp_forward(mlp, bc_right);
        Tensor *bc_left_loss = tensor_mse(bc_left_pred, bc_zero_left);
        Tensor *bc_right_loss = tensor_mse(bc_right_pred, bc_zero_right);
        Tensor *bc_loss = tensor_add(bc_left_loss, bc_right_loss);
        Tensor *bc_ic_loss = tensor_add(bc_loss, ic_loss);
        Tensor *total_loss = tensor_add(loss, bc_ic_loss);

        backward(total_loss);
        adam_step(adam);

        if(i % 100 == 0 || i == n_steps - 1){
            printf("step=%d physics=%f ic=%f bc=%f total=%f\n", i, loss->data[0], ic_loss->data[0], bc_loss->data[0], total_loss->data[0]);
            fprintf(loss_file, "%d,%f,%f,%f,%f\n", i, loss->data[0], ic_loss->data[0], bc_loss->data[0], total_loss->data[0]);
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
    Tensor *eval_pred = mlp_forward(mlp, eval_points);
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
    tensor_free(bc_zero_right);
    tensor_free(bc_zero_left);
    tensor_free(bc_right);
    tensor_free(bc_left);
    tensor_free(ic_targets);
    tensor_free(ic_points);
    adam_free(adam);
    free(params);
    mlp_free(mlp);
    return 0;
}
