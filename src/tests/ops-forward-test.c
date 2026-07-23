#include <math.h>
#include <stdio.h>

#include "pinn/core/backend.h"
#include "pinn/core/ops.h"
#include "pinn/core/tensor.h"

static int check_tensor(Tensor *tensor, const float *expected, int size, const char *name){
    if(!tensor){
        printf("%s failed: NULL tensor\n", name);
        return 0;
    }

    tensor_to_cpu(tensor);
    for(int i = 0; i < size; i++){
        if(fabsf(tensor->data[i] - expected[i]) > 1e-4f){
            printf("%s failed at %d: actual=%f expected=%f\n", name, i, tensor->data[i], expected[i]);
            return 0;
        }
    }
    return 1;
}

static int run_suite(Device device){
    int ok = 1;

    int shape[1] = {4};
    float a_values[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    float b_values[4] = {10.0f, 20.0f, -30.0f, 40.0f};
    float add_expected[4] = {11.0f, 18.0f, -27.0f, 44.0f};
    float sub_expected[4] = {-9.0f, -22.0f, 33.0f, -36.0f};
    float mul_expected[4] = {10.0f, -40.0f, -90.0f, 160.0f};
    float square_expected[4] = {1.0f, 4.0f, 9.0f, 16.0f};
    float scalar_mult_expected[4] = {2.0f, -4.0f, 6.0f, 8.0f};
    float scalar_add_expected[4] = {3.0f, 0.0f, 5.0f, 6.0f};
    float identity_expected[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    float relu_expected[4] = {1.0f, 0.0f, 3.0f, 4.0f};
    float sigmoid_expected[4] = {
        0.7310586f, 0.1192029f, 0.9525741f, 0.9820138f
    };
    float tanh_expected[4] = {
        0.7615942f, -0.9640276f, 0.9950548f, 0.9993293f
    };
    float mse_expected[1] = {737.5f};

    Tensor *a = tensor_from_data_device(a_values, shape, 1, 0, device);
    Tensor *b = tensor_from_data_device(b_values, shape, 1, 0, device);
    Tensor *out = NULL;

    out = tensor_add(a, b);
    ok &= check_tensor(out, add_expected, 4, "add");
    tensor_free(out);
    out = tensor_sub(a, b);
    ok &= check_tensor(out, sub_expected, 4, "sub");
    tensor_free(out);
    out = tensor_mult(a, b);
    ok &= check_tensor(out, mul_expected, 4, "mul");
    tensor_free(out);
    out = tensor_square(a);
    ok &= check_tensor(out, square_expected, 4, "square");
    tensor_free(out);
    out = tensor_tanh(a);
    ok &= check_tensor(out, tanh_expected, 4, "tanh");
    tensor_free(out);
    out = tensor_scalar_mult(a, 2.0f);
    ok &= check_tensor(out, scalar_mult_expected, 4, "scalar_mult");
    tensor_free(out);
    out = tensor_scalar_add(a, 2.0f);
    ok &= check_tensor(out, scalar_add_expected, 4, "scalar_add");
    tensor_free(out);
    out = tensor_identity(a);
    ok &= check_tensor(out, identity_expected, 4, "identity");
    tensor_free(out);
    out = tensor_relu(a);
    ok &= check_tensor(out, relu_expected, 4, "relu");
    tensor_free(out);
    out = tensor_sigmoid(a);
    ok &= check_tensor(out, sigmoid_expected, 4, "sigmoid");
    tensor_free(out);

    float mean_expected[1] = {1.5f};
    out = tensor_mean(a);
    ok &= check_tensor(out, mean_expected, 1, "mean");
    tensor_free(out);
    out = tensor_mse(a, b);
    ok &= check_tensor(out, mse_expected, 1, "mse");
    tensor_free(out);

    int mat_a_shape[2] = {2, 3};
    int mat_b_shape[2] = {3, 2};
    int bias_shape[1] = {2};
    float mat_a_values[6] = {1, 2, 3, 4, 5, 6};
    float mat_b_values[6] = {7, 8, 9, 10, 11, 12};
    float bias_values[2] = {100, 200};
    float mat_expected[4] = {58, 64, 139, 154};
    float bias_expected[4] = {158, 264, 239, 354};

    Tensor *mat_a = tensor_from_data_device(mat_a_values, mat_a_shape, 2, 0, device);
    Tensor *mat_b = tensor_from_data_device(mat_b_values, mat_b_shape, 2, 0, device);
    Tensor *bias = tensor_from_data_device(bias_values, bias_shape, 1, 0, device);
    Tensor *mat_out = tensor_matmult(mat_a, mat_b);
    Tensor *biased = tensor_bias_add(mat_out, bias);
    ok &= check_tensor(mat_out, mat_expected, 4, "matmult");
    ok &= check_tensor(biased, bias_expected, 4, "bias_add");

    tensor_free(biased);
    tensor_free(mat_out);
    tensor_free(bias);
    tensor_free(mat_b);
    tensor_free(mat_a);
    tensor_free(b);
    tensor_free(a);

    return ok;
}

static int run_structured_suite(Device device){
    int ok = 1;

    int d1_shape[2] = {2, 6};
    float d1_values[12] = {
        10, 11, 12,
        20, 21, 22,
        30, 31, 32,
        40, 41, 42
    };
    float select_d1_expected[4] = {11, 21, 31, 41};
    Tensor *d1 = tensor_from_data_device(d1_values, d1_shape, 2, 0, device);
    Tensor *selected_d1 = tensor_select_d1(d1, 3, 1);
    ok &= check_tensor(selected_d1, select_d1_expected, 4, "select_d1");

    int d2_shape[2] = {2, 8};
    float d2_values[16] = {
        100, 101, 102, 103,
        110, 111, 112, 113,
        200, 201, 202, 203,
        210, 211, 212, 213
    };
    float select_d2_expected[4] = {103, 113, 203, 213};
    Tensor *d2 = tensor_from_data_device(d2_values, d2_shape, 2, 0, device);
    Tensor *selected_d2 = tensor_select_d2(d2, 2, 1, 1);
    ok &= check_tensor(selected_d2, select_d2_expected, 4, "select_d2");

    int col_shape[2] = {3, 2};
    float col_values[6] = {
        0, 100,
        0.5f, 110,
        1, 120
    };
    float select_col_expected[3] = {100, 110, 120};
    Tensor *col = tensor_from_data_device(col_values, col_shape, 2, 0, device);
    Tensor *selected_col = tensor_select_col(col, 1);
    ok &= check_tensor(selected_col, select_col_expected, 3, "select_col");

    int factor_shape[1] = {2};
    float factor_values[2] = {2, 3};
    float scale_expected[6] = {2, 4, 6, 12, 15, 18};
    float scale_d1_values[6] = {1, 2, 3, 4, 5, 6};
    int scale_d1_shape[2] = {2, 3};
    Tensor *scale_d1 = tensor_from_data_device(scale_d1_values, scale_d1_shape, 2, 0, device);
    Tensor *factor = tensor_from_data_device(factor_values, factor_shape, 1, 0, device);
    Tensor *scaled = tensor_scale_deriv(scale_d1, factor, 3);
    ok &= check_tensor(scaled, scale_expected, 6, "scale_deriv");

    int chain_d1_shape[2] = {2, 2};
    int chain_d2_shape[2] = {2, 4};
    int scalar_shape[1] = {2};
    float chain_d1_values[4] = {1, 2, 3, 4};
    float chain_d2_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float f_prime_values[2] = {2, 3};
    float f_double_values[2] = {4, 5};
    float chain_expected[8] = {6, 12, 14, 24, 60, 78, 81, 104};
    Tensor *chain_d1 = tensor_from_data_device(chain_d1_values, chain_d1_shape, 2, 0, device);
    Tensor *chain_d2 = tensor_from_data_device(chain_d2_values, chain_d2_shape, 2, 0, device);
    Tensor *f_prime = tensor_from_data_device(f_prime_values, scalar_shape, 1, 0, device);
    Tensor *f_double = tensor_from_data_device(f_double_values, scalar_shape, 1, 0, device);
    Tensor *chained = tensor_chain_d2(chain_d1, chain_d2, f_prime, f_double, 2);
    ok &= check_tensor(chained, chain_expected, 8, "chain_d2");

    int w_shape[2] = {2, 2};
    float w_values[4] = {1, 2, 3, 4};
    Tensor *W = tensor_from_data_device(w_values, w_shape, 2, 0, device);

    int deriv1_shape[2] = {4, 2};
    float deriv1_values[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float deriv1_expected[8] = {
        10, 14, 14, 20,
        26, 30, 38, 44
    };
    Tensor *deriv1 = tensor_from_data_device(deriv1_values, deriv1_shape, 2, 0, device);
    Tensor *deriv1_out = tensor_deriv_matmult(deriv1, W, 2, 2, 2, 2);
    ok &= check_tensor(deriv1_out, deriv1_expected, 8, "deriv_matmult");

    int deriv2_shape[2] = {4, 4};
    float deriv2_values[16] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    float deriv2_expected[16] = {
        16, 20, 24, 28,
        22, 28, 34, 40,
        48, 52, 56, 60,
        70, 76, 82, 88
    };
    Tensor *deriv2 = tensor_from_data_device(deriv2_values, deriv2_shape, 2, 0, device);
    Tensor *deriv2_out = tensor_deriv2_matmult(deriv2, W, 2, 2, 2, 2);
    ok &= check_tensor(deriv2_out, deriv2_expected, 16, "deriv2_matmult");

    tensor_free(deriv2_out);
    tensor_free(deriv2);
    tensor_free(deriv1_out);
    tensor_free(deriv1);
    tensor_free(W);
    tensor_free(chained);
    tensor_free(f_double);
    tensor_free(f_prime);
    tensor_free(chain_d2);
    tensor_free(chain_d1);
    tensor_free(scaled);
    tensor_free(factor);
    tensor_free(scale_d1);
    tensor_free(selected_col);
    tensor_free(col);
    tensor_free(selected_d2);
    tensor_free(d2);
    tensor_free(selected_d1);
    tensor_free(d1);

    return ok;
}

int main(void){
    int ok = 1;

    printf("CPU forward suite\n");
    ok &= run_suite(DEVICE_CPU);
    printf("CPU structured forward suite\n");
    ok &= run_structured_suite(DEVICE_CPU);

#ifdef PINN_USE_CUDA
    if(backend_cuda_available()){
        printf("CUDA forward suite\n");
        ok &= run_suite(DEVICE_CUDA);
        printf("CUDA structured forward suite\n");
        ok &= run_structured_suite(DEVICE_CUDA);
    }
    else {
        printf("CUDA forward suite skipped: no CUDA device available\n");
    }
#else
    printf("CUDA forward suite skipped: CUDA not compiled\n");
#endif

    if(!ok){
        printf("forward operation tests failed\n");
        return 1;
    }

    printf("all forward operation tests passed\n");
    return 0;
}
