# Minimum CMake required
cmake_minimum_required(VERSION 3.1)

# Project
project(onnx-xla C CXX)

# Set C++11 as standard for the whole project
set(CMAKE_CXX_STANDARD 11)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -O0 -ggdb")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -ggdb")

include_directories(
  "${PROJECT_SOURCE_DIR}")

# Build the libraries with -fPIC
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED ONNX_NAMESPACE)
  set(ONNX_NAMESPACE onnx_xla)
endif()
add_definitions("-DONNX_NAMESPACE=${ONNX_NAMESPACE}")

# onnx library
add_subdirectory(${PROJECT_SOURCE_DIR}/third_party/onnx)

# xla library
SET(XLA_INCLUDE_DIRS
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow"
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-genfiles"
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow/external/protobuf_archive/src"
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-tensorflow/external/eigen_archive")
SET(XLA_LIBRARIES
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-bin/tensorflow/compiler/tf2xla/lib/libutil.so"
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-bin/tensorflow/compiler/xla/rpc/libxla_computation_client.so"
  "${PROJECT_SOURCE_DIR}/third_party/tensorflow/bazel-bin/tensorflow/libtensorflow_framework.so")
add_custom_command(
  OUTPUT ${XLA_LIBRARIES}
  COMMAND ${PROJECT_SOURCE_DIR}/scripts/build_xla.sh)
add_custom_target(xla
  SOURCES ${XLA_LIBRARIES})

# TODO: Get this from grpc CMake files
SET(GRPC_LIBRARIES
  "/usr/local/lib/libgrpc.so"
  "/usr/local/lib/libgrpc++.so")

# onnx_xla library (whole archive when anything links to it)
 file(GLOB_RECURSE onnx_xla_src
  onnx_xla/*.cc)
add_library(onnx_xla ${onnx_xla_src})
add_dependencies(onnx_xla xla)
target_link_libraries(onnx_xla onnx ${XLA_LIBRARIES} ${GRPC_LIBRARIES})
target_include_directories(onnx_xla PUBLIC ${XLA_INCLUDE_DIRS})

# python interface to ONNXIFI

if("${PY_EXT_SUFFIX}" STREQUAL "")
  if (MSVC)
    set(PY_EXT_SUFFIX ".pyd")
  else()
    set(PY_EXT_SUFFIX ".so")
  endif()
endif()

add_library(python_onnxifi MODULE "${PROJECT_SOURCE_DIR}/python_onnxifi/python_onnxifi.cc"
                                  "${PROJECT_SOURCE_DIR}/python_onnxifi/data_conversion.cc")
set_target_properties(python_onnxifi PROPERTIES PREFIX "")
set_target_properties(python_onnxifi PROPERTIES COMPILE_FLAGS "-fvisibility=hidden")
set_target_properties(python_onnxifi PROPERTIES SUFFIX ${PY_EXT_SUFFIX})
set_target_properties(python_onnxifi PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})
target_include_directories(python_onnxifi PRIVATE "${PYTHON_INCLUDE_DIR}" ${PROJECT_SOURCE_DIR}/third_party/onnx)

find_package(pybind11)
if(pybind11_FOUND)
  target_include_directories(python_onnxifi PRIVATE ${pybind11_INCLUDE_DIRS})
else()
  target_include_directories(python_onnxifi PRIVATE ${PROJECT_SOURCE_DIR}/third_party/pybind11/include)
endif()

target_link_libraries(python_onnxifi PRIVATE -Wl,--whole-archive onnx_xla -Wl,--no-whole-archive)

# binaries
file(GLOB_RECURSE binaries
  bin/*.cc)
foreach(bin ${binaries})
  get_filename_component(name ${bin} NAME_WE)
  add_executable(${name} ${bin})
  target_link_libraries(${name} -Wl,--whole-archive onnx_xla -Wl,--no-whole-archive)
endforeach()
