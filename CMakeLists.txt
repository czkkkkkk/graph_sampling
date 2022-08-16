cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(GS LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 14)

file(TO_NATIVE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/scripts/find_torch.py FIND_TORCH_PY)
set(PYTHON_INTERP python)
message(STATUS "Using Python interpreter: ${PYTHON_INTERP}")
execute_process(
  COMMAND ${PYTHON_INTERP} ${FIND_TORCH_PY}
  OUTPUT_VARIABLE TORCH_PREFIX_VER
  OUTPUT_STRIP_TRAILING_WHITESPACE)
message(STATUS "find_torch.py output: ${TORCH_PREFIX_VER}")
list(GET TORCH_PREFIX_VER 0 TORCH_PREFIX)
list(GET TORCH_PREFIX_VER 1 TORCH_VER)
message(STATUS "Configuring for PyTorch ${TORCH_VER}")
set(Torch_DIR "${TORCH_PREFIX}/Torch")
find_package(Torch REQUIRED)

file(GLOB GS_SRC
  src/*.cc
  src/cuda/*.cu
  src/cuda/*.cu.h
)

add_library(gs SHARED ${GS_SRC})
target_link_libraries(gs "${TORCH_LIBRARIES}")
set_target_properties(gs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(gs PROPERTIES CUDA_STANDARD 14)
set_target_properties(gs PROPERTIES POSITION_INDEPENDENT_CODE ON)
