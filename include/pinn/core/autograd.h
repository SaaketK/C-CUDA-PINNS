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
    void *ctx;
    void (*free_ctx)(void*);
} Node;

void backward(Tensor *loss);

// Node Free using Tape/
typedef struct {
    Node **nodes; // all autograd nodes created by ops
    Tensor **tensors; // all output tensors created by ops
    int n_nodes;
    int node_capacity;
    int n_tensors;
    int tensor_capacity;
} Tape;

Tape* tape_create(void);
int tape_add_node(Tape *tape, Node *node);
int tape_add_tensor(Tape *tape, Tensor *tensor);
Tape* get_curr_tape();
void set_curr_tape(Tape *tape);
void tape_free(Tape *tape);


// Nodelist to hold an array of nodes
typedef struct {
    Node **items;
    int size;
    int capacity;
} NodeList;

void nodelist_init(NodeList *list);
int nodelist_push(NodeList *list, Node *node);
int nodelist_contains(NodeList *list, Node *node);
void nodelist_free(NodeList *list);
// void build_topo(Tensor *t, NodeList *topo);

// Context structs that can be accessed by ops that need to preserve contextual values

typedef struct {
    float scalar;
} ScalarCtx;

typedef struct {
    int input_dim;
} ScaleDerivCtx;

typedef struct {
    int input_dim;
} ChainD2Ctx;

typedef struct {
    int batch;
    int in_features;
    int out_features;
    int input_dim;
    int order;
} DerivMatmultCtx;

typedef struct {
    int input_dim;
    int component;
} SelectD1Ctx;

typedef struct {
    int input_dim;
    int p;
    int q;
} SelectD2Ctx;

#endif
