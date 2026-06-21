/*
 * examples/blackscholes1d/blackscholes1d.c
 *
 * Training loop for 1D Black-Scholes equation. The PDE is:
 * V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0 
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

static float normal_cdf(float x){
    return 0.5f * (1.0f + erff(x / sqrtf(2.0f)));
}

static float black_scholes1d_exact(float t, float S, BlackScholes1DParams *params){
    float r = params->r;
    float K = params->K;
    float T = params->T;
    float sigma = params->sigma;
    float tau = T - t;

    if(tau <= 0.0f) return fmaxf(S - K, 0.0f);
    if(S <= 0.0f) return 0.0f;

    float sqrt_tau = sqrtf(tau);
    float d1 = (logf(S / K) + (r + 0.5f * sigma * sigma) * tau) / (sigma * sqrt_tau);
    float d2 = d1 - sigma * sqrt_tau;

    return S * normal_cdf(d1) - K * expf(-r * tau) * normal_cdf(d2);
}

int main(void) {
    int n_steps = 2500;
    int n_col = 200;
    int n_terminal = 100;
    int n_bc = 100;
    int sizes[] = {2, 64, 64, 64, 1};

    MLP *mlp = mlp_create(sizes, 5);

    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Adam *adam = adam_create(params, n_params, 1e-3f);

    BlackScholes1DParams bs = {
        .r = 0.05f,
        .K = 100.0f,
        .T = 1.0f,
        .sigma = 0.2f,
        .S_max = 200.0f,
        .payoff_beta = 0.25f
    };
    float S_max = bs.S_max;


    float lower[] = {0.0f, 0.0f};
    float upper[] = {bs.T, S_max};

    BoxDomain domain = {.dim = 2, .lower = lower, .upper = upper};

    Tensor *terminal_points = sample_fixed_dim_box(&domain, n_terminal, 0, bs.T);
    int terminal_shape[] = {n_terminal, 1};
    Tensor *terminal_targets = tensor_create(terminal_shape, 2, 0);

    for(int i = 0; i < n_terminal; i++){
        float S = terminal_points->data[i * 2 + 1];
        terminal_targets->data[i] = fmaxf(S - bs.K, 0.0f);
    }

    Tensor *bc_left = sample_fixed_dim_box(&domain, n_bc, 1, 0.0f);
    Tensor *bc_right = sample_fixed_dim_box(&domain, n_bc, 1, S_max);
    int bc_shape[] = {n_bc, 1};
    Tensor *bc_left_targets = tensor_create(bc_shape, 2, 0);
    Tensor *bc_right_targets = tensor_create(bc_shape, 2, 0);
    
    for(int i = 0; i < n_bc; i++){
        float t = bc_right->data[i * 2];
        float tau = bs.T - t;
        bc_right_targets->data[i] = S_max - bs.K * expf(-bs.r * tau);
    }

    FILE *loss_file = fopen("examples/blackscholes1d/files/blackscholes1d_loss.csv", "w");
    if(!loss_file){
        printf("failed to open examples/blackscholes1d/files/blackscholes1d_loss.csv\n");
        return 1;
    }
    fprintf(loss_file, "step,physics,terminal,bc,total,weighted_physics,weighted_terminal,weighted_bc\n");

    float physics_weight = 1.0f;
    float terminal_weight = 0.0f;
    float bc_weight = 0.0f;

    clock_t start = clock();
    printf("Hyperparameters: n_steps=%d n_col=%d n_terminal=%d n_bc=%d\n", n_steps, n_col, n_terminal, n_bc);
    for(int i = 0; i < n_steps; i++){ // Epochs
        Tape *tape = tape_create();
        set_curr_tape(tape);
        JetTape *jet_tape = jet_tape_create();
        set_curr_jet_tape(jet_tape);

        adam_zero_grad(adam);
        Tensor *points = sample_uniform_box(&domain, n_col);
        JetTensor *xj = jet_create_input(points, 2);
        JetTensor *N = jet_mlp_forward(mlp, xj);
        Tensor *residual = black_scholes1d_residual(N, points, &bs);
        Tensor *loss = residual_mse_loss(residual);
        Tensor *terminal_raw = mlp_forward(mlp, terminal_points);
        Tensor *terminal_pred = black_scholes1d_ansatz(terminal_raw, terminal_points, &bs);
        Tensor *terminal_loss = tensor_mse(terminal_pred, terminal_targets);
        Tensor *bc_left_raw = mlp_forward(mlp, bc_left);
        Tensor *bc_right_raw = mlp_forward(mlp, bc_right);
        Tensor *bc_left_pred = black_scholes1d_ansatz(bc_left_raw, bc_left, &bs);
        Tensor *bc_right_pred = black_scholes1d_ansatz(bc_right_raw, bc_right, &bs);
        Tensor *bc_left_loss = tensor_mse(bc_left_pred, bc_left_targets);
        Tensor *bc_right_loss = tensor_mse(bc_right_pred, bc_right_targets);
        Tensor *bc_loss = tensor_add(bc_left_loss, bc_right_loss);
        Tensor *weighted_phys_loss = tensor_scalar_mult(loss, physics_weight);
        Tensor *weighted_terminal_loss = tensor_scalar_mult(terminal_loss, terminal_weight);
        Tensor *weighted_bc_loss = tensor_scalar_mult(bc_loss, bc_weight);
        Tensor *weighted_bc_terminal_loss = tensor_add(weighted_terminal_loss, weighted_bc_loss);
        Tensor *total_loss = tensor_add(weighted_phys_loss, weighted_bc_terminal_loss);

        backward(total_loss);
        adam_step(adam);

        if(i % 100 == 0 || i == n_steps - 1){
            printf(
                "step=%d physics=%f terminal=%f bc=%f total=%f\n",
                i,
                loss->data[0],
                terminal_loss->data[0],
                bc_loss->data[0],
                total_loss->data[0]
            );
            fprintf(
                loss_file,
                "%d,%f,%f,%f,%f,%f,%f,%f\n",
                i,
                loss->data[0],
                terminal_loss->data[0],
                bc_loss->data[0],
                total_loss->data[0],
                weighted_phys_loss->data[0],
                weighted_terminal_loss->data[0],
                weighted_bc_loss->data[0]
            );
        }

        jet_tape_free_shallow(jet_tape);
        jet_free(xj);
        tape_free(tape);
    }

    double elapsed = (double)(clock() - start) / CLOCKS_PER_SEC;
    printf("training seconds=%f\n", elapsed);
    fclose(loss_file);

    int n_t_eval = 101;
    int n_S_eval = 101;
    int n_eval = n_t_eval * n_S_eval;
    int eval_shape[2] = {n_eval, 2};
    Tensor *eval_points = tensor_create(eval_shape, 2, 0);

    for(int i = 0; i < n_t_eval; i++){
        float t = bs.T * (float)i / (float)(n_t_eval - 1);
        for(int j = 0; j < n_S_eval; j++){
            float S = S_max * (float)j / (float)(n_S_eval - 1);
            int idx = i * n_S_eval + j;
            eval_points->data[idx * 2] = t;
            eval_points->data[idx * 2 + 1] = S;
        }
    }

    Tape *eval_tape = tape_create();
    set_curr_tape(eval_tape);
    clock_t infer_start = clock();
    Tensor *eval_raw = mlp_forward(mlp, eval_points);
    Tensor *eval_pred = black_scholes1d_ansatz(eval_raw, eval_points, &bs);
    double pinn_inference_seconds = (double)(clock() - infer_start) / CLOCKS_PER_SEC;

    FILE *pred_file = fopen("examples/blackscholes1d/files/blackscholes1d_predictions.csv", "w");
    if(!pred_file){
        printf("failed to open examples/blackscholes1d/files/blackscholes1d_predictions.csv\n");
        tape_free(eval_tape);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(pred_file, "t,S,V_pred,V_exact\n");

    for(int i = 0; i < n_t_eval; i++){
        for(int j = 0; j < n_S_eval; j++){
            int idx = i * n_S_eval + j;
            float t = eval_points->data[idx * 2];
            float S = eval_points->data[idx * 2 + 1];
            float V_pred = eval_pred->data[idx];
            float V_exact = black_scholes1d_exact(t, S, &bs);
            fprintf(pred_file, "%f,%f,%f,%f\n", t, S, V_pred, V_exact);
        }
    }

    fclose(pred_file);
    FILE *metrics_file = fopen("examples/blackscholes1d/files/blackscholes1d_metrics.csv", "w");
    if(!metrics_file){
        printf("failed to open examples/blackscholes1d/files/blackscholes1d_metrics.csv\n");
        tape_free(eval_tape);
        tensor_free(eval_points);
        return 1;
    }
    fprintf(metrics_file, "metric,value\n");
    fprintf(metrics_file, "training_seconds,%f\n", elapsed);
    fprintf(metrics_file, "pinn_inference_seconds,%f\n", pinn_inference_seconds);
    fprintf(metrics_file, "eval_points,%d\n", n_eval);
    fprintf(metrics_file, "S_max,%f\n", S_max);
    fprintf(metrics_file, "strike,%f\n", bs.K);
    fprintf(metrics_file, "maturity,%f\n", bs.T);
    fprintf(metrics_file, "rate,%f\n", bs.r);
    fprintf(metrics_file, "sigma,%f\n", bs.sigma);
    fclose(metrics_file);
    printf("wrote examples/blackscholes1d/files/blackscholes1d_loss.csv\n");
    printf("wrote examples/blackscholes1d/files/blackscholes1d_predictions.csv\n");
    printf("wrote examples/blackscholes1d/files/blackscholes1d_metrics.csv\n");
    tape_free(eval_tape);
    tensor_free(eval_points);
    tensor_free(bc_right_targets);
    tensor_free(bc_left_targets);
    tensor_free(bc_right);
    tensor_free(bc_left);
    tensor_free(terminal_targets);
    tensor_free(terminal_points);
    adam_free(adam);
    free(params);
    mlp_free(mlp);
    return 0;
}
