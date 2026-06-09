/*
 * core/autograd.h
 *
 * Owns the autograd graph API: Node definition, backward entry point, and later
 * graph utilities such as topological traversal, gradient seeding, zeroing, and
 * graph cleanup declarations.
 */

#ifndef PINN_CORE_AUTOGRAD_H
#define PINN_CORE_AUTOGRAD_H

#include "pinn/core/tensor.h"

// Node to represent operations in the computational graph.
typedef struct Node {
    Tensor **inputs;
    int n_inputs; // Number of inputs; eg. 2 for add, 2 for mul
    Tensor *output;
    void (*backward)(struct Node*);
} Node;

void backward(Tensor *loss);

#endif
