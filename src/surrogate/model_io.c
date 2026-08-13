#include "pinn/surrogate/model_io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char MODEL_MAGIC[8] = {'P', 'I', 'N', 'N', 'M', 'D', 'L', '1'};

static const uint32_t MODEL_VERSION = 1;

static int write_data(FILE *file, const void *data, size_t size){
    return fwrite(data, 1, size, file) == size;
}

static int read_data(FILE *file, void *data, size_t size){
    return fread(data, 1, size, file) == size;
}

int pinn_model_save(const char *path, const MLP *mlp, const float *input_lower, const float *input_upper){
    if(!path || !mlp || !input_lower || !input_upper || mlp->n_layers <= 0) return 0;

    FILE *file = fopen(path, "wb");
    if(!file) return 0;

    uint32_t n_layers = (uint32_t)mlp->n_layers;
    uint32_t input_dim = (uint32_t)mlp->layers[0]->input_dim;

    int ok = write_data(file, MODEL_MAGIC, sizeof(MODEL_MAGIC))
        && write_data(file, &MODEL_VERSION, sizeof(MODEL_VERSION))
        && write_data(file, &n_layers, sizeof(n_layers))
        && write_data(file, &input_dim, sizeof(input_dim))
        && write_data(file, input_lower, input_dim * sizeof(float))
        && write_data(file, input_upper, input_dim * sizeof(float));
    
    for(int i = 0; ok && i < mlp->n_layers; i++){
        Linear *layer = mlp->layers[i];
        uint32_t in_dim = (uint32_t)layer->input_dim;
        uint32_t out_dim = (uint32_t)layer->output_dim;

        ok = write_data(file, &in_dim, sizeof(in_dim))
            && write_data(file, &out_dim, sizeof(out_dim))
            && write_data(file, layer->W->data, layer->W->size * sizeof(float))
            && write_data(file, layer->b->data, layer->b->size * sizeof(float));
    }

    fclose(file);
    return ok;
}

typedef struct {
    uint32_t input_dim;
    uint32_t output_dim;
    float *weights;
    float *biases;
} SerializedLayer;

MLP* pinn_model_load(const char *path, float **input_lower_out, float **input_upper_out, ActivationFn activation_fn){
    if(!path || !input_lower_out || !input_upper_out) return 0;

    FILE *file = fopen(path, "rb");
    if(!file) return 0;

    *input_lower_out = NULL;
    *input_upper_out = NULL;

    unsigned char magic[sizeof(MODEL_MAGIC)];
    uint32_t version = 0;
    uint32_t n_layers = 0;
    uint32_t input_dim = 0;

    float *input_lower = NULL;
    float *input_upper = NULL;
    int *layer_sizes = NULL;
    SerializedLayer *saved_layers = NULL;
    MLP *mlp = NULL;

    int ok = read_data(file, magic, sizeof(magic))
        && memcmp(magic, MODEL_MAGIC, sizeof(MODEL_MAGIC)) == 0
        && read_data(file, &version, sizeof(version))
        && version == MODEL_VERSION
        && read_data(file, &n_layers, sizeof(n_layers))
        && read_data(file, &input_dim, sizeof(input_dim));

    if(!ok || n_layers == 0 || n_layers > 64 || input_dim == 0) goto fail;

    input_lower = malloc(input_dim * sizeof(float));
    input_upper = malloc(input_dim * sizeof(float));
    layer_sizes = calloc(n_layers + 1, sizeof(int));
    saved_layers = calloc(n_layers, sizeof(SerializedLayer));

    if(!input_lower || !input_upper || !layer_sizes || !saved_layers) goto fail;
    if(!read_data(file, input_lower, input_dim * sizeof(float)) || !read_data(file, input_upper, input_dim * sizeof(float))) goto fail;

    for(uint32_t i = 0; i < n_layers; i++){
        SerializedLayer *layer = &saved_layers[i];

        if(!read_data(file, &layer->input_dim, sizeof(layer->input_dim))
        || !read_data(file, &layer->output_dim, sizeof(layer->output_dim))
        || layer->input_dim == 0
        || layer->output_dim == 0){
            goto fail;
        }

        if(i == 0){
            if(layer->input_dim != input_dim) goto fail;
            layer_sizes[0] = (int)layer->input_dim;
        }
        else if(layer_sizes[i] != (int)layer->input_dim){
            goto fail;
        }
        layer_sizes[i + 1] = (int)layer->output_dim;

        size_t weight_ct = (size_t)layer->input_dim * layer->output_dim;
        layer->weights = malloc(weight_ct * sizeof(float));
        layer->biases = malloc(layer->output_dim * sizeof(float));
        
        if(!layer->weights || !layer->biases || !read_data(file, layer->weights, weight_ct * sizeof(float)) || !read_data(file, layer->biases, layer->output_dim * sizeof(float))) goto fail;
    }

    mlp = mlp_create(layer_sizes, (int)n_layers + 1, activation_fn);
    if(!mlp) goto fail;

    for(uint32_t i = 0; i < n_layers; i++){
        Linear *layer = mlp->layers[i];
        SerializedLayer *saved = &saved_layers[i];

        memcpy(layer->W->data, saved->weights, layer->W->size * sizeof(float));
        memcpy(layer->b->data, saved->biases, layer->b->size * sizeof(float));
    }

    for(uint32_t i = 0; i < n_layers; i++){
        free(saved_layers[i].weights);
        free(saved_layers[i].biases);
    }

    free(saved_layers);
    free(layer_sizes);
    fclose(file);

    *input_lower_out = input_lower;
    *input_upper_out = input_upper;
    return mlp;

    fail:
        if(mlp) mlp_free(mlp);
        if(saved_layers){
            for(uint32_t i = 0; i < n_layers; i++){
                free(saved_layers[i].weights);
                free(saved_layers[i].biases);
            }
        }
        free(saved_layers);
        free(layer_sizes);
        free(input_lower);
        free(input_upper);
        fclose(file);
        return NULL;
}