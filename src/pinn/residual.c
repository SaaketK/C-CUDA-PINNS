/*
 * pinn/residual.c
 *
 * Will implement equation-agnostic residual helpers and constraint-loss
 * utilities. Equation-specific callbacks, such as heat or Black-Scholes
 * residuals, will use JetTensor derivatives to form physics losses while the
 * trainer handles sampling and optimization.
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "pinn/pinn/residual.h"
#include "pinn/autodiff/jet.h"
#include "pinn/core/autograd.h"
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"

static Tensor* residual_const_tensor(const int *shape, int ndim){
    Tensor *tensor = tensor_create(shape, ndim, 0);
    Tape *tape = get_curr_tape();
    if(tape){
        tape_add_tensor(tape, tensor);
    }
    return tensor;
}

static float black_scholes1d_smax(BlackScholes1DParams *params){
    return params->S_max > 0.0f ? params->S_max : 2.0f * params->K;
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

static void black_scholes1d_payoff(
    float S,
    BlackScholes1DParams *params,
    float *payoff,
    float *payoff_S,
    float *payoff_SS
){
    float beta = params->payoff_beta;
    if(beta <= 0.0f){
        *payoff = fmaxf(S - params->K, 0.0f);
        *payoff_S = S > params->K ? 1.0f : 0.0f;
        *payoff_SS = 0.0f;
        return;
    }

    float z = beta * (S - params->K);
    float sig = stable_sigmoid(z);
    *payoff = stable_softplus(z) / beta;
    *payoff_S = sig;
    *payoff_SS = beta * sig * (1.0f - sig);
}

Tensor* heat1d_residual(JetTensor *u, Tensor *points, Heat1DParams *params){
    (void)points;
    Tensor *u_t = tensor_select_d1(u->d1, u->input_dim, 0);
    Tensor *u_xx = tensor_select_d2(u->d2, u->input_dim, 1, 1);
    Tensor *alpha_u_xx = tensor_scalar_mult(u_xx, params->alpha);
    Tensor *residual = tensor_sub(u_t, alpha_u_xx);
    return residual;
}

Tensor* heat1d_ansatz_residual(JetTensor *N, Tensor *points, Heat1DParams *params){
    // u(t,x) = (1-t) * sin(pi*x) + t*x*(1-x) * N(t,x).
    // This exactly enforces u(0,x)=sin(pi*x), u(t,0)=0, and u(t,1)=0.
    Tensor *N_t = tensor_select_d1(N->d1, N->input_dim, 0);
    Tensor *N_x = tensor_select_d1(N->d1, N->input_dim, 1);
    Tensor *N_xx = tensor_select_d2(N->d2, N->input_dim, 1, 1);

    int n = points->shape[0];
    int shape[] = {n, 1};
    Tensor *A_t = residual_const_tensor(shape, 2);
    Tensor *A_xx = residual_const_tensor(shape, 2);
    Tensor *B = residual_const_tensor(shape, 2);
    Tensor *B_t = residual_const_tensor(shape, 2);
    Tensor *B_x = residual_const_tensor(shape, 2);
    Tensor *B_xx = residual_const_tensor(shape, 2);

    for(int i = 0; i < n; i++){
        float t = points->data[i * 2];
        float x = points->data[i * 2 + 1];
        float sin_px = sinf((float)M_PI * x);

        A_t->data[i] = -sin_px;
        A_xx->data[i] = -(1.0f - t) * (float)M_PI * (float)M_PI * sin_px;
        B->data[i] = t * x * (1.0f - x);
        B_t->data[i] = x * (1.0f - x);
        B_x->data[i] = t * (1.0f - 2.0f * x);
        B_xx->data[i] = -2.0f * t;
    }

    Tensor *u_t = tensor_add(A_t, tensor_add(tensor_mult(B_t, N->value), tensor_mult(B, N_t)));
    Tensor *u_xx = tensor_add(
        A_xx,
        tensor_add(
            tensor_mult(B_xx, N->value),
            tensor_add(
                tensor_scalar_mult(tensor_mult(B_x, N_x), 2.0f),
                tensor_mult(B, N_xx)
            )
        )
    );
    Tensor *alpha_u_xx = tensor_scalar_mult(u_xx, params->alpha);
    return tensor_sub(u_t, alpha_u_xx);
}

Tensor* heat1d_ansatz(Tensor *raw, Tensor *points, Heat1DParams *params){
    (void)params;
    int n = points->shape[0];
    int shape[] = {n, 1};
    Tensor *A = residual_const_tensor(shape, 2);
    Tensor *B = residual_const_tensor(shape, 2);

    for(int i = 0; i < n; i++){
        float t = points->data[i * 2];
        float x = points->data[i * 2 + 1];
        A->data[i] = (1.0f - t) * sinf((float)M_PI * x);
        B->data[i] = t * x * (1.0f - x);
    }

    return tensor_add(A, tensor_mult(B, raw));
}

Tensor* heat2d_ansatz_residual(JetTensor *N, Tensor *points, Heat2DParams *params){
    // u(t,x,y) = (1-t) sin(pi x) sin(pi y) + t x(1-x) y(1-y) N(t,x,y).
    // This exactly enforces u(0,x,y)=sin(pi x)sin(pi y) and u=0 on all
    // four spatial boundary faces.
    Tensor *N_t = tensor_select_d1(N->d1, N->input_dim, 0);
    Tensor *N_x = tensor_select_d1(N->d1, N->input_dim, 1);
    Tensor *N_xx = tensor_select_d2(N->d2, N->input_dim, 1, 1);
    Tensor *N_y = tensor_select_d1(N->d1, N->input_dim, 2);
    Tensor *N_yy = tensor_select_d2(N->d2, N->input_dim, 2, 2);

    int n = points->shape[0];
    int shape[] = {n, 1};
    Tensor *A_t = residual_const_tensor(shape, 2);
    Tensor *A_xx = residual_const_tensor(shape, 2);
    Tensor *A_yy = residual_const_tensor(shape, 2);
    Tensor *B = residual_const_tensor(shape, 2);
    Tensor *B_t = residual_const_tensor(shape, 2);
    Tensor *B_x = residual_const_tensor(shape, 2);
    Tensor *B_xx = residual_const_tensor(shape, 2);
    Tensor *B_y = residual_const_tensor(shape, 2);
    Tensor *B_yy = residual_const_tensor(shape, 2);

    for(int i = 0; i < n; i++){
        float t = points->data[i * 3];
        float x = points->data[i * 3 + 1];
        float y = points->data[i * 3 + 2];
        float sin_px = sinf((float)M_PI * x);
        float sin_py = sinf((float)M_PI * y);
        float x_factor = x * (1.0f - x);
        float y_factor = y * (1.0f - y);

        A_t->data[i] = -sin_px * sin_py;
        A_xx->data[i] = -(1.0f - t) * (float)M_PI * (float)M_PI * sin_px * sin_py;
        A_yy->data[i] = A_xx->data[i];
        B->data[i] = t * x_factor * y_factor;
        B_t->data[i] = x_factor * y_factor;
        B_x->data[i] = t * (1.0f - 2.0f * x) * y_factor;
        B_xx->data[i] = -2.0f * t * y_factor;
        B_y->data[i] = t * x_factor * (1.0f - 2.0f * y);
        B_yy->data[i] = -2.0f * t * x_factor;
    }

    Tensor *u_t = tensor_add(A_t, tensor_add(tensor_mult(B_t, N->value), tensor_mult(B, N_t)));
    Tensor *u_xx = tensor_add(
        A_xx,
        tensor_add(
            tensor_mult(B_xx, N->value),
            tensor_add(
                tensor_scalar_mult(tensor_mult(B_x, N_x), 2.0f),
                tensor_mult(B, N_xx)
            )
        )
    );
    Tensor *u_yy = tensor_add(
        A_yy,
        tensor_add(
            tensor_mult(B_yy, N->value),
            tensor_add(
                tensor_scalar_mult(tensor_mult(B_y, N_y), 2.0f),
                tensor_mult(B, N_yy)
            )
        )
    );
    Tensor *alpha_x_u_xx = tensor_scalar_mult(u_xx, params->alpha_x);
    Tensor *alpha_y_u_yy = tensor_scalar_mult(u_yy, params->alpha_y);
    return tensor_sub(u_t, tensor_add(alpha_x_u_xx, alpha_y_u_yy));
}

Tensor* heat2d_ansatz(Tensor *raw, Tensor *points, Heat2DParams *params){
    (void)params;
    int n = points->shape[0];
    int shape[] = {n, 1};
    Tensor *A = residual_const_tensor(shape, 2);
    Tensor *B = residual_const_tensor(shape, 2);

    for(int i = 0; i < n; i++){
        float t = points->data[i * 3];
        float x = points->data[i * 3 + 1];
        float y = points->data[i * 3 + 2];
        A->data[i] = (1.0f - t) * sinf((float)M_PI * x) * sinf((float)M_PI * y);
        B->data[i] = t * x * (1.0f - x) * y * (1.0f - y);
    }

    return tensor_add(A, tensor_mult(B, raw));
}

Tensor* residual_mse_loss(Tensor *residual){
    Tensor *square = tensor_square(residual);
    return tensor_mean(square);
}

Tensor* black_scholes1d_residual(JetTensor *N, Tensor *points, BlackScholes1DParams *params){
    // Boundary ansatz:
    // V(t,S) = A(t,S) + B(t,S) * N(t,S)
    // A enforces terminal, S=0, and S=S_max constraints.
    // B = (T-t) * s * (1-s) vanishes on those constrained edges.
    // PDE: V_t + 0.5 * sigma^2 * S^2 * V_SS + r * S * V_S - r * V = 0.
    Tensor *N_t = tensor_select_d1(N->d1, N->input_dim, 0);
    Tensor *N_S = tensor_select_d1(N->d1, N->input_dim, 1);
    Tensor *N_SS = tensor_select_d2(N->d2, N->input_dim, 1, 1);
    Tensor *S = tensor_select_col(points, 1);

    int n = points->shape[0];
    int shape[] = {n, 1};
    Tensor *A = residual_const_tensor(shape, 2);
    Tensor *A_t = residual_const_tensor(shape, 2);
    Tensor *A_S = residual_const_tensor(shape, 2);
    Tensor *A_SS = residual_const_tensor(shape, 2);
    Tensor *B = residual_const_tensor(shape, 2);
    Tensor *B_t = residual_const_tensor(shape, 2);
    Tensor *B_S = residual_const_tensor(shape, 2);
    Tensor *B_SS = residual_const_tensor(shape, 2);

    float S_max = black_scholes1d_smax(params);
    float payoff_max;
    float payoff_max_S;
    float payoff_max_SS;
    black_scholes1d_payoff(S_max, params, &payoff_max, &payoff_max_S, &payoff_max_SS);

    for(int i = 0; i < n; i++){
        float t = points->data[i * 2];
        float Si = points->data[i * 2 + 1];
        float tau = params->T - t;
        float s = Si / S_max;
        float one_minus_s = 1.0f - s;
        float exp_term = expf(-params->r * tau);
        float payoff;
        float payoff_S;
        float payoff_SS;
        float right = S_max - params->K * exp_term;
        float C = right - payoff_max;
        float C_t = -params->r * params->K * exp_term;
        black_scholes1d_payoff(Si, params, &payoff, &payoff_S, &payoff_SS);

        A->data[i] = payoff + s * C;
        A_t->data[i] = s * C_t;
        A_S->data[i] = payoff_S + C / S_max;
        A_SS->data[i] = payoff_SS;
        B->data[i] = tau * s * one_minus_s;
        B_t->data[i] = -s * one_minus_s;
        B_S->data[i] = tau * (1.0f - 2.0f * s) / S_max;
        B_SS->data[i] = -2.0f * tau / (S_max * S_max);
    }

    Tensor *V = tensor_add(A, tensor_mult(B, N->value));
    Tensor *V_t = tensor_add(A_t, tensor_add(tensor_mult(B_t, N->value), tensor_mult(B, N_t)));
    Tensor *V_S = tensor_add(A_S, tensor_add(tensor_mult(B_S, N->value), tensor_mult(B, N_S)));
    Tensor *V_SS = tensor_add(
        A_SS,
        tensor_add(
            tensor_mult(B_SS, N->value),
            tensor_add(
                tensor_scalar_mult(tensor_mult(B_S, N_S), 2.0f),
                tensor_mult(B, N_SS)
            )
        )
    );

    // let w = 0.5 * sigma^2 * S^2 
    Tensor *w = tensor_scalar_mult(tensor_scalar_mult(tensor_square(S), params->sigma * params->sigma), 0.5f);
    Tensor *w_V_SS = tensor_mult(w, V_SS);
    Tensor *r_S_V_S = tensor_scalar_mult(tensor_mult(S, V_S), params->r);
    Tensor *r_V = tensor_scalar_mult(V, params->r);
    Tensor *residual = tensor_add(tensor_add(V_t, w_V_SS), tensor_sub(r_S_V_S,r_V));
    return residual;
}

Tensor* black_scholes1d_ansatz(Tensor *raw, Tensor *points, BlackScholes1DParams *params){
    // Boundary ansatz: V(t,S) = A(t,S) + (T-t) * s * (1-s) * raw(t,S).
    int n = points->shape[0];
    int shape[] = {n, 1};
    Tensor *A = residual_const_tensor(shape, 2);
    Tensor *B = residual_const_tensor(shape, 2);

    float S_max = black_scholes1d_smax(params);
    float payoff_max;
    float payoff_max_S;
    float payoff_max_SS;
    black_scholes1d_payoff(S_max, params, &payoff_max, &payoff_max_S, &payoff_max_SS);

    for(int i = 0; i < n; i++){
        float t = points->data[i * 2];
        float Si = points->data[i * 2 + 1];
        float tau = params->T - t;
        float s = Si / S_max;
        float payoff;
        float payoff_S;
        float payoff_SS;
        float right = S_max - params->K * expf(-params->r * tau);
        float C = right - payoff_max;
        black_scholes1d_payoff(Si, params, &payoff, &payoff_S, &payoff_SS);

        A->data[i] = payoff + s * C;
        B->data[i] = tau * s * (1.0f - s);
    }

    return tensor_add(A, tensor_mult(B, raw));
}
