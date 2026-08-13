/*
 * src/tests/core-test.c
 *
 * Smoke tests for tensor storage, core tensor ops, and reverse-mode autograd.
*/

#include <stdio.h>
#include <math.h>
#include "pinn/core/tensor.h"
#include "pinn/core/ops.h"
#include "pinn/core/autograd.h"

static int check_close(const char *name, float actual, float expected){
    float err = fabsf(actual - expected);
    printf("%s actual=%f expected=%f abs_err=%f\n", name, actual, expected, err);
    return err < 1e-5f;
}

int main(void) {
    int shape[1] = {3};
    float values[3] = {1.0f, 2.0f, 3.0f};

    Tensor *x = tensor_from_data(values, shape, 1, 1);

    tensor_print(x, "x");

    tensor_fill(x, 5.0f);
    tensor_print(x, "filled x");

    tensor_zero(x);
    tensor_print(x, "zeroed x");

    tensor_zero_grad(x);
    tensor_free(x);

    float a_values[3] = {1.0f, 2.0f, 3.0f};
    float b_values[3] = {10.0f, 20.0f, 30.0f};

    Tensor *a = tensor_from_data(a_values, shape, 1, 1);
    Tensor *b = tensor_from_data(b_values, shape, 1, 1);

    Tensor *sum = tensor_add(a, b);
    Tensor *prod = tensor_mult(a, b);

    tensor_print(sum, "a + b");
    tensor_print(prod, "a * b");

    Tensor *sq = tensor_square(a);
    Tensor *mean_sq = tensor_mean(sq);
    Tensor *mse = tensor_mse(a, b);

    tensor_print(sq, "a squared");
    tensor_print(mean_sq, "mean(a squared)");
    tensor_print(mse, "mse(a, b)");

    int mat_a_shape[2] = {2, 3};
    int mat_b_shape[2] = {3, 2};
    int bias_shape[1] = {2};

    float mat_a_values[6] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    };
    float mat_b_values[6] = {
        7.0f, 8.0f,
        9.0f, 10.0f,
        11.0f, 12.0f
    };
    float bias_values[2] = {100.0f, 200.0f};

    Tensor *mat_a = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    Tensor *mat_b = tensor_from_data(mat_b_values, mat_b_shape, 2, 1);
    Tensor *bias = tensor_from_data(bias_values, bias_shape, 1, 1);

    Tensor *mat_prod = tensor_matmult(mat_a, mat_b);
    Tensor *biased = tensor_bias_add(mat_prod, bias);

    tensor_print(mat_prod, "mat_a @ mat_b");
    tensor_print(biased, "(mat_a @ mat_b) + bias");

    Tensor *ag_x = tensor_from_data(values, shape, 1, 1);
    Tensor *ag_sq = tensor_mult(ag_x, ag_x);
    Tensor *ag_loss = tensor_mean(ag_sq);

    backward(ag_loss);

    tensor_print(ag_loss, "mean(ag_x * ag_x)");
    printf("ag_x grad=[%f, %f, %f]\n", ag_x->grad[0], ag_x->grad[1], ag_x->grad[2]);

    Tensor *add_x = tensor_from_data(values, shape, 1, 1);
    Tensor *add_y = tensor_add(add_x, add_x);
    Tensor *add_loss = tensor_mean(add_y);
    backward(add_loss);
    tensor_print(add_loss, "mean(add_x + add_x)");
    printf("add_x grad=[%f, %f, %f]\n", add_x->grad[0], add_x->grad[1], add_x->grad[2]);

    Tensor *sub_x = tensor_from_data(values, shape, 1, 1);
    Tensor *sub_y = tensor_from_data(b_values, shape, 1, 1);
    Tensor *sub_out = tensor_sub(sub_x, sub_y);
    Tensor *sub_loss = tensor_mean(sub_out);
    backward(sub_loss);
    tensor_print(sub_loss, "mean(sub_x - sub_y)");
    printf("sub_x grad=[%f, %f, %f]\n", sub_x->grad[0], sub_x->grad[1], sub_x->grad[2]);
    printf("sub_y grad=[%f, %f, %f]\n", sub_y->grad[0], sub_y->grad[1], sub_y->grad[2]);

    Tensor *square_x = tensor_from_data(values, shape, 1, 1);
    Tensor *square_y = tensor_square(square_x);
    Tensor *square_loss = tensor_mean(square_y);
    backward(square_loss);
    tensor_print(square_loss, "mean(square_x squared)");
    printf("square_x grad=[%f, %f, %f]\n", square_x->grad[0], square_x->grad[1], square_x->grad[2]);

    Tensor *tanh_x = tensor_from_data(values, shape, 1, 1);
    Tensor *tanh_y = tensor_tanh(tanh_x);
    Tensor *tanh_loss = tensor_mean(tanh_y);
    backward(tanh_loss);
    tensor_print(tanh_loss, "mean(tanh(tanh_x))");
    printf("tanh_x grad=[%f, %f, %f]\n", tanh_x->grad[0], tanh_x->grad[1], tanh_x->grad[2]);

    Tensor *mm_a = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    Tensor *mm_b = tensor_from_data(mat_b_values, mat_b_shape, 2, 1);
    Tensor *mm_prod = tensor_matmult(mm_a, mm_b);
    Tensor *mm_loss = tensor_mean(mm_prod);
    backward(mm_loss);
    tensor_print(mm_loss, "mean(mm_a @ mm_b)");
    printf("mm_a grad=[%f, %f, %f, %f, %f, %f]\n",
           mm_a->grad[0], mm_a->grad[1], mm_a->grad[2],
           mm_a->grad[3], mm_a->grad[4], mm_a->grad[5]);
    printf("mm_b grad=[%f, %f, %f, %f, %f, %f]\n",
           mm_b->grad[0], mm_b->grad[1], mm_b->grad[2],
           mm_b->grad[3], mm_b->grad[4], mm_b->grad[5]);

    int ba_bias_shape[1] = {3};
    float ba_bias_values[3] = {10.0f, 20.0f, 30.0f};
    Tensor *ba_a = tensor_from_data(mat_a_values, mat_a_shape, 2, 1);
    Tensor *ba_b = tensor_from_data(ba_bias_values, ba_bias_shape, 1, 1);
    Tensor *ba_out = tensor_bias_add(ba_a, ba_b);
    Tensor *ba_loss = tensor_mean(ba_out);
    backward(ba_loss);
    tensor_print(ba_loss, "mean(ba_a + ba_b)");
    printf("ba_a grad=[%f, %f, %f, %f, %f, %f]\n",
           ba_a->grad[0], ba_a->grad[1], ba_a->grad[2],
           ba_a->grad[3], ba_a->grad[4], ba_a->grad[5]);
    printf("ba_b grad=[%f, %f, %f]\n", ba_b->grad[0], ba_b->grad[1], ba_b->grad[2]);

    Tensor *mse_x = tensor_from_data(values, shape, 1, 1);
    Tensor *mse_y = tensor_from_data(b_values, shape, 1, 1);
    Tensor *mse_loss = tensor_mse(mse_x, mse_y);
    backward(mse_loss);
    tensor_print(mse_loss, "mse(mse_x, mse_y)");
    printf("mse_x grad=[%f, %f, %f]\n", mse_x->grad[0], mse_x->grad[1], mse_x->grad[2]);
    printf("mse_y grad=[%f, %f, %f]\n", mse_y->grad[0], mse_y->grad[1], mse_y->grad[2]);

    int deriv_shape[2] = {2, 2};
    int factor_shape[1] = {2};
    float deriv_values[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float factor_values[2] = {10.0f, 20.0f};
    Tensor *scale_deriv_x = tensor_from_data(deriv_values, deriv_shape, 2, 1);
    Tensor *scale_factor = tensor_from_data(factor_values, factor_shape, 1, 1);
    Tensor *scale_out = tensor_scale_deriv(scale_deriv_x, scale_factor, 2);
    Tensor *scale_loss = tensor_mean(scale_out);
    backward(scale_loss);

    int scale_ok = 1;
    scale_ok &= check_close("scale_deriv out[0]", scale_out->data[0], 10.0f);
    scale_ok &= check_close("scale_deriv out[1]", scale_out->data[1], 20.0f);
    scale_ok &= check_close("scale_deriv out[2]", scale_out->data[2], 60.0f);
    scale_ok &= check_close("scale_deriv out[3]", scale_out->data[3], 80.0f);
    scale_ok &= check_close("scale_deriv deriv grad[0]", scale_deriv_x->grad[0], 2.5f);
    scale_ok &= check_close("scale_deriv deriv grad[1]", scale_deriv_x->grad[1], 2.5f);
    scale_ok &= check_close("scale_deriv deriv grad[2]", scale_deriv_x->grad[2], 5.0f);
    scale_ok &= check_close("scale_deriv deriv grad[3]", scale_deriv_x->grad[3], 5.0f);
    scale_ok &= check_close("scale_deriv factor grad[0]", scale_factor->grad[0], 0.75f);
    scale_ok &= check_close("scale_deriv factor grad[1]", scale_factor->grad[1], 1.75f);
    if(!scale_ok){
        printf("scale_deriv test failed\n");
        return 1;
    }
    printf("scale_deriv test passed\n");

    int d1_shape[2] = {2, 6};
    float d1_values[12] = {
        10.0f, 11.0f, 12.0f,
        20.0f, 21.0f, 22.0f,
        30.0f, 31.0f, 32.0f,
        40.0f, 41.0f, 42.0f
    };
    Tensor *select_d1_x = tensor_from_data(d1_values, d1_shape, 2, 1);
    Tensor *select_d1_out = tensor_select_d1(select_d1_x, 3, 1);
    Tensor *select_d1_loss = tensor_mean(select_d1_out);
    backward(select_d1_loss);

    int select_d1_ok = 1;
    select_d1_ok &= check_close("select_d1 out[0]", select_d1_out->data[0], 11.0f);
    select_d1_ok &= check_close("select_d1 out[1]", select_d1_out->data[1], 21.0f);
    select_d1_ok &= check_close("select_d1 out[2]", select_d1_out->data[2], 31.0f);
    select_d1_ok &= check_close("select_d1 out[3]", select_d1_out->data[3], 41.0f);
    select_d1_ok &= check_close("select_d1 grad[0]", select_d1_x->grad[0], 0.0f);
    select_d1_ok &= check_close("select_d1 grad[1]", select_d1_x->grad[1], 0.25f);
    select_d1_ok &= check_close("select_d1 grad[2]", select_d1_x->grad[2], 0.0f);
    select_d1_ok &= check_close("select_d1 grad[3]", select_d1_x->grad[3], 0.0f);
    select_d1_ok &= check_close("select_d1 grad[4]", select_d1_x->grad[4], 0.25f);
    select_d1_ok &= check_close("select_d1 grad[5]", select_d1_x->grad[5], 0.0f);
    select_d1_ok &= check_close("select_d1 grad[7]", select_d1_x->grad[7], 0.25f);
    select_d1_ok &= check_close("select_d1 grad[10]", select_d1_x->grad[10], 0.25f);
    if(!select_d1_ok){
        printf("select_d1 test failed\n");
        return 1;
    }
    printf("select_d1 test passed\n");

    int d2_shape[2] = {2, 8};
    float d2_values[16] = {
        100.0f, 101.0f, 102.0f, 103.0f,
        110.0f, 111.0f, 112.0f, 113.0f,
        200.0f, 201.0f, 202.0f, 203.0f,
        210.0f, 211.0f, 212.0f, 213.0f
    };
    Tensor *select_d2_x = tensor_from_data(d2_values, d2_shape, 2, 1);
    Tensor *select_d2_out = tensor_select_d2(select_d2_x, 2, 1, 1);
    Tensor *select_d2_loss = tensor_mean(select_d2_out);
    backward(select_d2_loss);

    int select_d2_ok = 1;
    select_d2_ok &= check_close("select_d2 out[0]", select_d2_out->data[0], 103.0f);
    select_d2_ok &= check_close("select_d2 out[1]", select_d2_out->data[1], 113.0f);
    select_d2_ok &= check_close("select_d2 out[2]", select_d2_out->data[2], 203.0f);
    select_d2_ok &= check_close("select_d2 out[3]", select_d2_out->data[3], 213.0f);
    select_d2_ok &= check_close("select_d2 grad[0]", select_d2_x->grad[0], 0.0f);
    select_d2_ok &= check_close("select_d2 grad[3]", select_d2_x->grad[3], 0.25f);
    select_d2_ok &= check_close("select_d2 grad[7]", select_d2_x->grad[7], 0.25f);
    select_d2_ok &= check_close("select_d2 grad[8]", select_d2_x->grad[8], 0.0f);
    select_d2_ok &= check_close("select_d2 grad[11]", select_d2_x->grad[11], 0.25f);
    select_d2_ok &= check_close("select_d2 grad[15]", select_d2_x->grad[15], 0.25f);
    if(!select_d2_ok){
        printf("select_d2 test failed\n");
        return 1;
    }
    printf("select_d2 test passed\n");

    int select_col_shape[2] = {3, 2};
    float select_col_values[6] = {
        0.0f, 100.0f,
        0.5f, 110.0f,
        1.0f, 120.0f
    };
    Tensor *select_col_x = tensor_from_data(select_col_values, select_col_shape, 2, 1);
    Tensor *select_col_out = tensor_select_col(select_col_x, 1);
    Tensor *select_col_loss = tensor_mean(select_col_out);
    backward(select_col_loss);

    int select_col_ok = 1;
    select_col_ok &= check_close("select_col out[0]", select_col_out->data[0], 100.0f);
    select_col_ok &= check_close("select_col out[1]", select_col_out->data[1], 110.0f);
    select_col_ok &= check_close("select_col out[2]", select_col_out->data[2], 120.0f);
    select_col_ok &= check_close("select_col grad[0]", select_col_x->grad[0], 0.0f);
    select_col_ok &= check_close("select_col grad[1]", select_col_x->grad[1], 1.0f / 3.0f);
    select_col_ok &= check_close("select_col grad[2]", select_col_x->grad[2], 0.0f);
    select_col_ok &= check_close("select_col grad[3]", select_col_x->grad[3], 1.0f / 3.0f);
    select_col_ok &= check_close("select_col grad[4]", select_col_x->grad[4], 0.0f);
    select_col_ok &= check_close("select_col grad[5]", select_col_x->grad[5], 1.0f / 3.0f);
    if(!select_col_ok){
        printf("select_col test failed\n");
        return 1;
    }
    printf("select_col test passed\n");

    tensor_free(select_col_loss);
    tensor_free(select_col_out);
    tensor_free(select_col_x);
    tensor_free(select_d2_loss);
    tensor_free(select_d2_out);
    tensor_free(select_d2_x);
    tensor_free(select_d1_loss);
    tensor_free(select_d1_out);
    tensor_free(select_d1_x);
    tensor_free(scale_loss);
    tensor_free(scale_out);
    tensor_free(scale_factor);
    tensor_free(scale_deriv_x);
    tensor_free(mse_loss);
    tensor_free(mse_y);
    tensor_free(mse_x);
    tensor_free(ba_loss);
    tensor_free(ba_out);
    tensor_free(ba_b);
    tensor_free(ba_a);
    tensor_free(mm_loss);
    tensor_free(mm_prod);
    tensor_free(mm_b);
    tensor_free(mm_a);
    tensor_free(tanh_loss);
    tensor_free(tanh_y);
    tensor_free(tanh_x);
    tensor_free(square_loss);
    tensor_free(square_y);
    tensor_free(square_x);
    tensor_free(sub_loss);
    tensor_free(sub_out);
    tensor_free(sub_y);
    tensor_free(sub_x);
    tensor_free(add_loss);
    tensor_free(add_y);
    tensor_free(add_x);
    tensor_free(ag_loss);
    tensor_free(ag_sq);
    tensor_free(ag_x);
    tensor_free(sq);
    tensor_free(mean_sq);
    tensor_free(mse);
    tensor_free(mat_prod);
    tensor_free(biased);
    tensor_free(mat_a);
    tensor_free(mat_b);
    tensor_free(bias);
    tensor_free(sum);
    tensor_free(prod);
    tensor_free(a);
    tensor_free(b);

    return 0;
}
