/*
 * core/autograd.h
 *
 * Owns the autograd graph API: Node definition, backward entry point, and later
 * graph utilities such as topological traversal, gradient seeding, zeroing, and
 * graph cleanup declarations.
 */

 #include <stdio.h>
 #include <stdlib.h>
 #include "tensor.h"


 // Node to represent operations in the computational graph
typedef struct Node {
    Tensor **inputs;
    int n_inputs; // Number of inputs; eg. 2 for add, 2 for mul
    Tensor *output;
    void (*backward)(struct Node*);
} Node;
