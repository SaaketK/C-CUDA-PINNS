/*
 * CPU/CUDA reverse-mode parity test.  The Python runner in this directory
 * reads the BASIC rows below and compares them with PyTorch.
 */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "pinn/core/autograd.h"
#include "pinn/core/backend.h"
#include "pinn/core/ops.h"
#include "pinn/core/tensor.h"

#define BASIC_COUNT 21

static void copy_grad(float *dst, Tensor *tensor){
    tensor_to_cpu(tensor);
    memcpy(dst, tensor->grad, (size_t)tensor->size * sizeof(float));
}

static void print_values(const char *label, const float *values, int count){
    printf("%s", label);
    for(int i = 0; i < count; i++){
        printf(",%.9g", values[i]);
    }
    printf("\n");
}

static int run_basic(Device device, float *result){
    int x_shape[2] = {2, 3};
    int w_shape[2] = {3, 2};
    int b_shape[1] = {2};
    int target_shape[2] = {2, 2};
    float x_values[6] = {-1.0f, -0.5f, 0.25f, 0.75f, 1.25f, 1.5f};
    float y_values[6] = {0.5f, -1.5f, 2.0f, -0.25f, 0.75f, -1.0f};
    float w_values[6] = {0.2f, -0.3f, 0.4f, 0.1f, -0.5f, 0.6f};
    float b_values[2] = {0.15f, -0.2f};
    float target_values[4] = {0.1f, 0.9f, 0.3f, 0.7f};

    Tensor *x = tensor_from_data_device(x_values, x_shape, 2, 1, device);
    Tensor *y = tensor_from_data_device(y_values, x_shape, 2, 1, device);
    Tensor *w = tensor_from_data_device(w_values, w_shape, 2, 1, device);
    Tensor *b = tensor_from_data_device(b_values, b_shape, 1, 1, device);
    Tensor *target = tensor_from_data_device(target_values, target_shape, 2, 0, device);
    if(!x || !y || !w || !b || !target) return 0;

    Tape *tape = tape_create();
    set_curr_tape(tape);
    Tensor *product = tensor_mult(x, y);
    Tensor *activated = tensor_tanh(product);
    Tensor *linear = tensor_matmult(activated, w);
    Tensor *biased = tensor_bias_add(linear, b);
    Tensor *relu = tensor_relu(biased);
    Tensor *prediction = tensor_sigmoid(relu);
    Tensor *loss = tensor_mse(prediction, target);
    backward(loss);

    tensor_to_cpu(loss);
    result[0] = loss->data[0];
    copy_grad(result + 1, x);
    copy_grad(result + 7, y);
    copy_grad(result + 13, w);
    copy_grad(result + 19, b);

    tape_free(tape);
    tensor_free(target);
    tensor_free(b);
    tensor_free(w);
    tensor_free(y);
    tensor_free(x);
    return 1;
}

static int compare(const char *name, const float *cpu, const float *cuda, int count){
    for(int i = 0; i < count; i++){
        if(fabsf(cpu[i] - cuda[i]) > 2e-4f){
            printf("%s mismatch at %d: cpu=%f cuda=%f\n", name, i, cpu[i], cuda[i]);
            return 0;
        }
    }
    return 1;
}

int main(void){
    float cpu[BASIC_COUNT];
    float cuda[BASIC_COUNT];

    if(!run_basic(DEVICE_CPU, cpu)){
        printf("failed to run CPU backward test\n");
        return 1;
    }
    print_values("BASIC,CPU", cpu, BASIC_COUNT);

    if(!backend_cuda_available()){
        printf("CUDA unavailable; comparison skipped\n");
        return 0;
    }
    if(!run_basic(DEVICE_CUDA, cuda)){
        printf("failed to run CUDA backward test\n");
        return 1;
    }
    print_values("BASIC,CUDA", cuda, BASIC_COUNT);

    if(!compare("basic backward", cpu, cuda, BASIC_COUNT)){
        return 1;
    }
    printf("CPU/CUDA basic backward parity passed\n");
    return 0;
}
