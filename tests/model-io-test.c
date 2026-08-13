#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "pinn/core/tensor.h"
#include "pinn/nn/activations.h"
#include "pinn/nn/mlp.h"
#include "pinn/surrogate/model_io.h"

static int check_close(const char *name, float actual, float expected, float tol){
    float error = fabsf(actual - expected);
    if(!isfinite(actual) || error > tol){
        printf("%s FAILED: actual=%g expected=%g abs_error=%g\n",
            name, actual, expected, error);
        return 0;
    }
    return 1;
}

int main(void){
    const char *path = "/tmp/pinn-model-io-test.pinn";
    int sizes[] = {3, 4, 1};
    const float lower[] = {-1.0f, 0.0f, 0.1f};
    const float upper[] = {1.0f, 2.0f, 0.9f};
    const float inputs[] = {
        -0.25f, 0.5f, 0.2f,
         0.75f, 1.5f, 0.8f,
    };
    const int input_shape[] = {2, 3};

    MLP *original = mlp_create(sizes, 3, tanh_activation);
    for(int layer_index = 0; layer_index < original->n_layers; layer_index++){
        Linear *layer = original->layers[layer_index];
        for(int i = 0; i < layer->W->size; i++){
            layer->W->data[i] = 0.03125f * (float)(i + 1 + 10 * layer_index);
        }
        for(int i = 0; i < layer->b->size; i++){
            layer->b->data[i] = -0.0625f * (float)(i + 1 + layer_index);
        }
    }

    Tensor *input_before = tensor_from_data(inputs, input_shape, 2, 0);
    Tensor *output_before = mlp_forward(original, input_before);
    if(!pinn_model_save(path, original, lower, upper)){
        printf("model save FAILED\n");
        return 1;
    }

    float *loaded_lower = NULL;
    float *loaded_upper = NULL;
    MLP *loaded = pinn_model_load(path, &loaded_lower, &loaded_upper, tanh_activation);
    if(!loaded){
        printf("model load FAILED\n");
        return 1;
    }

    Tensor *input_after = tensor_from_data(inputs, input_shape, 2, 0);
    Tensor *output_after = mlp_forward(loaded, input_after);
    int ok = loaded->n_layers == original->n_layers;

    for(int i = 0; i < 3; i++){
        ok = check_close("input lower", loaded_lower[i], lower[i], 0.0f) && ok;
        ok = check_close("input upper", loaded_upper[i], upper[i], 0.0f) && ok;
    }

    for(int layer_index = 0; layer_index < original->n_layers; layer_index++){
        Linear *expected = original->layers[layer_index];
        Linear *actual = loaded->layers[layer_index];
        if(expected->input_dim != actual->input_dim
            || expected->output_dim != actual->output_dim){
            printf("layer %d dimension mismatch\n", layer_index);
            ok = 0;
            continue;
        }
        for(int i = 0; i < expected->W->size; i++){
            ok = check_close("weight", actual->W->data[i], expected->W->data[i], 0.0f) && ok;
        }
        for(int i = 0; i < expected->b->size; i++){
            ok = check_close("bias", actual->b->data[i], expected->b->data[i], 0.0f) && ok;
        }
    }

    for(int i = 0; i < output_before->size; i++){
        ok = check_close("prediction", output_after->data[i], output_before->data[i], 1e-6f) && ok;
    }

    tensor_free(output_after);
    tensor_free(input_after);
    mlp_free(loaded);
    free(loaded_lower);
    free(loaded_upper);
    tensor_free(output_before);
    tensor_free(input_before);
    mlp_free(original);
    remove(path);

    if(!ok) return 1;
    printf("model I/O round-trip test passed\n");
    return 0;
}
