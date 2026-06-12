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

add_executable(pinn_main
        src/main.c
        ${PINN_CORE_SOURCES})

add_executable(core_test
        src/tests/core-test.c
        ${PINN_CORE_SOURCES})

add_executable(nn_test
        src/tests/nn-test.c
        ${PINN_CORE_SOURCES}
        ${PINN_NN_SOURCES})

target_include_directories(pinn_main PRIVATE include)
target_include_directories(core_test PRIVATE include)
target_include_directories(nn_test PRIVATE include)
