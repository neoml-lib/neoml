cmake_minimum_required(VERSION 3.11 FATAL_ERROR)

project(NeoOnnx::Onnx2NeoML LANGUAGES CXX)

include(Version)
include(Settings)
include(GNUInstallDirs)

set_global_variables()

add_executable(Onnx2NeoML Onnx2NeoML.cpp)

target_link_libraries(Onnx2NeoML PRIVATE NeoML NeoOnnx NeoMathEngine)
