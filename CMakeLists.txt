cmake_minimum_required(VERSION 3.20)
project(PINN C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PINN_CORE_SOURCES
        src/core/tensor.c
        src/core/ops.c
        src/core/autograd.c)

set(PINN_NN_SOURCES
        src/nn/mlp.c
        src/nn/optimizer.c)

set(PINN_AUTODIFF_SOURCES
        src/autodiff/jet.c)

set(PINN_SAMPLER_SOURCES
        src/pinn/sampler.c)

set(PINN_RESIDUAL_SOURCES
        src/pinn/residual.c)

add_executable(pinn_main
        src/main.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(core_test
        src/tests/core-test.c
        ${PINN_CORE_SOURCES})

add_executable(nn_test
        src/tests/nn-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(gradcheck_test
        src/tests/gradcheck-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(jet_test
        src/tests/jet-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES})

add_executable(heat_residual_test
        src/tests/heat-residual-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(black_scholes_residual_test
        src/tests/black-scholes-residual-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(black_scholes1d_train
        examples/blackscholes1d/black_scholes1d.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES}
        ${PINN_AUTODIFF_SOURCES}
        ${PINN_SAMPLER_SOURCES}
        ${PINN_RESIDUAL_SOURCES})

add_executable(sampler_test
        src/tests/sampler-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_SAMPLER_SOURCES})

target_include_directories(pinn_main PRIVATE include)
target_include_directories(core_test PRIVATE include)
target_include_directories(nn_test PRIVATE include)
target_include_directories(gradcheck_test PRIVATE include)
target_include_directories(jet_test PRIVATE include)
target_include_directories(heat_residual_test PRIVATE include)
target_include_directories(black_scholes_residual_test PRIVATE include)
target_include_directories(black_scholes1d_train PRIVATE include)
target_include_directories(sampler_test PRIVATE include)
