/*
 * src/tests/jet-test.c
 *
 * Smoke tests for JetTensor forward-mode value, gradient, and Hessian
 * propagation.
 */

#include <math.h>
#include <stdio.h>
#include "pinn/autodiff/jet.h"
#include "pinn/core/autograd.h"
#include "pinn/core/ops.h"
#include "pinn/core/tensor.h"
#include "pinn/nn/mlp.h"

static int check_close(const char *name, float actual, float expected, float tol){
    float err = fabsf(actual - expected);
    printf("%s actual=%f expected=%f abs_err=%f\n", name, actual, expected, err);
    if(err > tol){
        printf("%s FAILED\n", name);
        return 0;
    }
    return 1;
}

static int test_create_input(void){
    int shape[2] = {1, 2};
    float values[2] = {2.0f, 3.0f};
    Tensor *value = tensor_from_data(values, shape, 2, 0);
    JetTensor *jet = jet_create_input(value, 2);
    int ok = 1;

    ok = check_close("input d t / d t", jet_get_d1(jet, 0, 0), 1.0f, 1e-6f) && ok;
    ok = check_close("input d t / d x", jet_get_d1(jet, 0, 1), 0.0f, 1e-6f) && ok;
    ok = check_close("input d x / d t", jet_get_d1(jet, 1, 0), 0.0f, 1e-6f) && ok;
    ok = check_close("input d x / d x", jet_get_d1(jet, 1, 1), 1.0f, 1e-6f) && ok;
    ok = check_close("input d2 t / d t d t", jet_get_d2(jet, 0, 0, 0), 0.0f, 1e-6f) && ok;
    ok = check_close("input d2 x / d x d x", jet_get_d2(jet, 1, 1, 1), 0.0f, 1e-6f) && ok;

    jet_free(jet);
    return ok;
}

static int test_square_single_var(void){
    int shape[2] = {1, 1};
    float values[1] = {3.0f};
    Tensor *value = tensor_from_data(values, shape, 2, 0);
    JetTensor *x = jet_create_input(value, 1);
    JetTensor *y = jet_square(x);
    int ok = 1;

    ok = check_close("square value", y->value->data[0], 9.0f, 1e-6f) && ok;
    ok = check_close("square d1", jet_get_d1(y, 0, 0), 6.0f, 1e-6f) && ok;
    ok = check_close("square d2", jet_get_d2(y, 0, 0, 0), 2.0f, 1e-6f) && ok;

    jet_free(y);
    jet_free(x);
    return ok;
}

static int test_add_two_var_squares(void){
    int shape[2] = {1, 2};
    float t_values[2] = {2.0f, 0.0f};
    float x_values[2] = {0.0f, 3.0f};
    Tensor *t_tensor = tensor_from_data(t_values, shape, 2, 0);
    Tensor *x_tensor = tensor_from_data(x_values, shape, 2, 0);
    JetTensor *t = jet_create(t_tensor, 2);
    JetTensor *x = jet_create(x_tensor, 2);
    jet_set_d1(t, 0, 0, 1.0f);
    jet_set_d1(x, 1, 1, 1.0f);

    JetTensor *t2 = jet_square(t);
    JetTensor *x2 = jet_square(x);
    JetTensor *f = jet_add(t2, x2);
    int ok = 1;

    ok = check_close("t2+x2 t slot value", f->value->data[0], 4.0f, 1e-6f) && ok;
    ok = check_close("t2+x2 x slot value", f->value->data[1], 9.0f, 1e-6f) && ok;
    ok = check_close("t2 d/dt", jet_get_d1(f, 0, 0), 4.0f, 1e-6f) && ok;
    ok = check_close("x2 d/dx", jet_get_d1(f, 1, 1), 6.0f, 1e-6f) && ok;
    ok = check_close("t2 d2/dtdt", jet_get_d2(f, 0, 0, 0), 2.0f, 1e-6f) && ok;
    ok = check_close("x2 d2/dxdx", jet_get_d2(f, 1, 1, 1), 2.0f, 1e-6f) && ok;
    ok = check_close("cross d2/dtdx", jet_get_d2(f, 0, 0, 1), 0.0f, 1e-6f) && ok;

    jet_free(f);
    jet_free(x2);
    jet_free(t2);
    jet_free(x);
    jet_free(t);
    return ok;
}

static int test_tanh_single_var(void){
    int shape[2] = {1, 1};
    float values[1] = {0.5f};
    Tensor *value = tensor_from_data(values, shape, 2, 0);
    JetTensor *x = jet_create_input(value, 1);
    JetTensor *y = jet_tanh(x);
    float yval = tanhf(0.5f);
    float d1 = 1.0f - yval * yval;
    float d2 = -2.0f * yval * d1;
    int ok = 1;

    ok = check_close("tanh value", y->value->data[0], yval, 1e-6f) && ok;
    ok = check_close("tanh d1", jet_get_d1(y, 0, 0), d1, 1e-6f) && ok;
    ok = check_close("tanh d2", jet_get_d2(y, 0, 0, 0), d2, 1e-6f) && ok;

    jet_free(y);
    jet_free(x);
    return ok;
}

static int test_tanh_d1_backward(void){
    int shape[2] = {1, 1};
    float values[1] = {0.5f};
    Tensor *value = tensor_from_data(values, shape, 2, 1);
    JetTensor *x = jet_create_input(value, 1);
    JetTensor *y = jet_tanh(x);
    Tensor *loss = tensor_mean(y->d1);
    float yval = tanhf(0.5f);
    float d1 = 1.0f - yval * yval;
    float expected_value_grad = -2.0f * yval * d1;
    int ok = 1;

    backward(loss);

    ok = check_close("tanh d1 backward value grad", value->grad[0], expected_value_grad, 1e-6f) && ok;
    ok = check_close("tanh d1 backward input d1 grad", x->d1->grad[0], d1, 1e-6f) && ok;

    tensor_free(loss);
    jet_free(y);
    jet_free(x);
    return ok;
}

static int test_matmult_bias_add(void){
    int x_shape[2] = {1, 2};
    int w_shape[2] = {2, 1};
    int b_shape[1] = {1};
    float x_values[2] = {2.0f, 3.0f};
    float w_values[2] = {4.0f, 5.0f};
    float b_values[1] = {7.0f};
    Tensor *x_tensor = tensor_from_data(x_values, x_shape, 2, 0);
    Tensor *W = tensor_from_data(w_values, w_shape, 2, 0);
    Tensor *b = tensor_from_data(b_values, b_shape, 1, 0);
    JetTensor *x = jet_create_input(x_tensor, 2);
    JetTensor *linear = jet_matmult(x, W);
    JetTensor *out = jet_bias_add(linear, b);
    int ok = 1;

    ok = check_close("linear+bias value", out->value->data[0], 30.0f, 1e-6f) && ok;
    ok = check_close("linear d/dx0", jet_get_d1(out, 0, 0), 4.0f, 1e-6f) && ok;
    ok = check_close("linear d/dx1", jet_get_d1(out, 0, 1), 5.0f, 1e-6f) && ok;
    ok = check_close("linear d2/dx0dx0", jet_get_d2(out, 0, 0, 0), 0.0f, 1e-6f) && ok;
    ok = check_close("linear d2/dx1dx1", jet_get_d2(out, 0, 1, 1), 0.0f, 1e-6f) && ok;

    jet_free(out);
    jet_free(linear);
    jet_free(x);
    tensor_free(b);
    tensor_free(W);
    return ok;
}

static int test_mlp_forward_jet_linear(void){
    int x_shape[2] = {1, 2};
    float x_values[2] = {4.0f, 5.0f};
    Tensor *x_tensor = tensor_from_data(x_values, x_shape, 2, 0);
    JetTensor *x = jet_create_input(x_tensor, 2);
    int sizes[2] = {2, 1};
    MLP *mlp = mlp_create(sizes, 2);
    mlp->layers[0]->W->data[0] = 2.0f;
    mlp->layers[0]->W->data[1] = 3.0f;
    mlp->layers[0]->b->data[0] = 1.0f;

    JetTensor *out = jet_mlp_forward(mlp, x);
    int ok = 1;

    ok = check_close("mlp jet linear value", out->value->data[0], 24.0f, 1e-6f) && ok;
    ok = check_close("mlp jet linear d/dt", jet_get_d1(out, 0, 0), 2.0f, 1e-6f) && ok;
    ok = check_close("mlp jet linear d/dx", jet_get_d1(out, 0, 1), 3.0f, 1e-6f) && ok;
    ok = check_close("mlp jet linear d2/dtdt", jet_get_d2(out, 0, 0, 0), 0.0f, 1e-6f) && ok;
    ok = check_close("mlp jet linear d2/dtdx", jet_get_d2(out, 0, 0, 1), 0.0f, 1e-6f) && ok;
    ok = check_close("mlp jet linear d2/dxdx", jet_get_d2(out, 0, 1, 1), 0.0f, 1e-6f) && ok;

    jet_free(out);
    jet_free(x);
    mlp_free(mlp);
    return ok;
}

static int test_jet_tape_mlp_forward(void){
    int x_shape[2] = {1, 2};
    float x_values[2] = {4.0f, 5.0f};
    Tensor *x_tensor = tensor_from_data(x_values, x_shape, 2, 0);
    JetTensor *x = jet_create_input(x_tensor, 2);
    int sizes[2] = {2, 1};
    MLP *mlp = mlp_create(sizes, 2);
    mlp->layers[0]->W->data[0] = 2.0f;
    mlp->layers[0]->W->data[1] = 3.0f;
    mlp->layers[0]->b->data[0] = 1.0f;

    JetTape *tape = jet_tape_create();
    set_curr_jet_tape(tape);
    JetTensor *out = jet_mlp_forward(mlp, x);
    int ok = 1;

    ok = check_close("jet tape mlp value", out->value->data[0], 24.0f, 1e-6f) && ok;
    ok = check_close("jet tape mlp d/dt", jet_get_d1(out, 0, 0), 2.0f, 1e-6f) && ok;
    ok = check_close("jet tape mlp d/dx", jet_get_d1(out, 0, 1), 3.0f, 1e-6f) && ok;
    if(tape->size != 2){
        printf("jet tape size FAILED actual=%d expected=2\n", tape->size);
        ok = 0;
    }

    jet_tape_free(tape);
    jet_free(x);
    mlp_free(mlp);
    return ok;
}

int main(void){
    int ok = 1;
    ok = test_create_input() && ok;
    ok = test_square_single_var() && ok;
    ok = test_add_two_var_squares() && ok;
    ok = test_tanh_single_var() && ok;
    ok = test_tanh_d1_backward() && ok;
    ok = test_matmult_bias_add() && ok;
    ok = test_mlp_forward_jet_linear() && ok;
    ok = test_jet_tape_mlp_forward() && ok;

    if(!ok){
        printf("JetTensor tests failed\n");
        return 1;
    }

    printf("all JetTensor tests passed\n");
    return 0;
}
