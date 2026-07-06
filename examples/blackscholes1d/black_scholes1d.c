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
#include "pinn/core/backend.h"

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
    backend_init();
    backend_set_default(backend_cuda_available() ? DEVICE_CUDA : DEVICE_CPU);
    Device device = backend_default_device();
    int n_steps = 2500;;
    int n_col = 500;
    int sizes[] = {2, 64, 64, 64, 1};

    MLP *mlp = mlp_create(sizes, 5, tanh_activation);

    int n_params = 0;
    Tensor **params = mlp_parameters(mlp, &n_params);
    Adam *adam = adam_create(params, n_params, 1e-3f);

    BlackScholes1DParams bs = {
        .r = 0.05f,
        .K = 100.0f,
        .T = 1.0f,
        .sigma = 0.2f,
        .S_max = 200.0f,
        .payoff_beta = 1.0f
    };
    float S_max = bs.S_max;


    float lower[] = {0.0f, 0.0f};
    float upper[] = {bs.T, S_max};

    BoxDomain domain = {.dim = 2, .lower = lower, .upper = upper};

    FILE *loss_file = fopen("examples/blackscholes1d/files/blackscholes1d_loss.csv", "w");
    if(!loss_file){
        printf("failed to open examples/blackscholes1d/files/blackscholes1d_loss.csv\n");
        return 1;
    }
    fprintf(loss_file, "step,physics,total\n");

    clock_t start = clock();
    printf("Hyperparameters: n_steps=%d n_col=%d sampler=LHS\n", n_steps, n_col);
    for(int i = 0; i < n_steps; i++){ // Epochs
        Tape *tape = tape_create();
        set_curr_tape(tape);
        JetTape *jet_tape = jet_tape_create();
        set_curr_jet_tape(jet_tape);

        adam_zero_grad(adam);
        Tensor *points = sample_LHS_box(&domain, n_col);
        JetTensor *xj = jet_create_input(points, 2);
        JetTensor *N = jet_mlp_forward(mlp, xj);
        Tensor *residual = black_scholes1d_residual(N, points, &bs);
        Tensor *loss = residual_mse_loss(residual);
        Tensor *total_loss = loss;

        backward(total_loss);
        adam_step(adam);

        if(i % 100 == 0 || i == n_steps - 1){
            printf(
                "step=%d physics=%f total=%f\n",
                i,
                loss->data[0],
                total_loss->data[0]
            );
            fprintf(
                loss_file,
                "%d,%f,%f\n",
                i,
                loss->data[0],
                total_loss->data[0]
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
    Tensor *eval_points = tensor_create(eval_shape, 2, 0, device);

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
    fprintf(metrics_file, "payoff_beta,%f\n", bs.payoff_beta);
    fclose(metrics_file);
    printf("wrote examples/blackscholes1d/files/blackscholes1d_loss.csv\n");
    printf("wrote examples/blackscholes1d/files/blackscholes1d_predictions.csv\n");
    printf("wrote examples/blackscholes1d/files/blackscholes1d_metrics.csv\n");
    tape_free(eval_tape);
    tensor_free(eval_points);
    adam_free(adam);
    free(params);
    mlp_free(mlp);
    return 0;
}
