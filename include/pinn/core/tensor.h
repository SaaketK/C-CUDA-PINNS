/*
 * core/tensor.h
 *
 * Owns the Tensor storage API: device tag, Tensor struct, shape/size metadata,
 * data and grad buffers, allocation/free helpers, zero/fill helpers, and debug
 * printing. It should not own operation definitions or graph traversal long-term.
 */

#ifndef TENSOR_H
#define TENSOR_H

typedef enum {
    DEVICE_CPU,
    DEVICE_CUDA,
} Device;

struct Node; // Forward Declaration

// Tensor to hold tensor data and gradient
typedef struct Tensor {
    float *data; // Tensor value
    float *grad; // gradient of the tensor
    int *shape;
    int ndim; 
    int size;
    int req_grad; // flag to indicate if this tensor requires gradient
    struct Node* grad_fn;
    Device device;
} Tensor;

// Function Declarations
Tensor* tensor_create(const int *shape, int ndim, int req_grad);
Tensor* tensor_from_data(const float *data, const int *shape, int ndim, int req_grad);
int tensor_size(const int *shape, int ndim); // Number of elements in tensor
void tensor_free(Tensor *tensor);
void tensor_zero(Tensor *tensor);
void tensor_zero_grad(Tensor *tensor);
void tensor_fill(Tensor *tensor, float value);
void tensor_print(Tensor *tensor, const char *name);

#endif
