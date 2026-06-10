/*
 * nn/activations.c
 *
 * Implements neural-network activation wrappers. These functions should call
 * differentiable core tensor ops such as tensor_tanh so activation layers remain
 * connected to the autograd graph.
 */
