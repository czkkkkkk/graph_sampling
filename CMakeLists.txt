cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

project(GS)
set(CMAKE_CXX_STANDARD 14)

# FIXME
set(Torch_DIR "/efs/zkcai/miniconda3/envs/gs/lib/python3.8/site-packages/torch/share/cmake/Torch")
find_package(Torch REQUIRED)

file(GLOB GS_SRC
  src/*.cc
)

add_library(gs SHARED ${GS_SRC})
target_link_libraries(gs "${TORCH_LIBRARIES}")