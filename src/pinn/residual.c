/*
 * pinn/residual.c
 *
 * Will implement equation-agnostic residual helpers and constraint-loss
 * utilities. Equation-specific callbacks, such as heat or Black-Scholes
 * residuals, will use JetTensor derivatives to form physics losses while the
 * trainer handles sampling and optimization.
*/
