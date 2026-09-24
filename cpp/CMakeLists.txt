cmake_minimum_required(VERSION 3.16)
project(aec_infer CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Where to find ONNX Runtime. Give it either as
#   cmake -DONNXRUNTIME_ROOT=/path/to/onnxruntime ..
# or through the environment:
#   ONNXRUNTIME_ROOT=/path/to/onnxruntime cmake ..
# The directory must contain include/ and lib/ (that is the layout of the
# official release archives, e.g. onnxruntime-linux-x64-1.2x.x).
if(NOT ONNXRUNTIME_ROOT)
  set(ONNXRUNTIME_ROOT "$ENV{ONNXRUNTIME_ROOT}")
endif()

if(ONNXRUNTIME_ROOT)
  message(STATUS "ONNX Runtime: ${ONNXRUNTIME_ROOT}")
  find_path(ORT_INCLUDE onnxruntime_cxx_api.h
            PATHS "${ONNXRUNTIME_ROOT}/include" NO_DEFAULT_PATH)
  find_library(ORT_LIB onnxruntime
               PATHS "${ONNXRUNTIME_ROOT}/lib" NO_DEFAULT_PATH)
else()
  message(STATUS "ONNX Runtime: searching the system")
  find_path(ORT_INCLUDE onnxruntime_cxx_api.h)
  find_library(ORT_LIB onnxruntime)
endif()

if(NOT ORT_INCLUDE OR NOT ORT_LIB)
  message(FATAL_ERROR
    "ONNX Runtime not found.\n"
    "  Download a release from https://github.com/microsoft/onnxruntime/releases\n"
    "  and re-run cmake with -DONNXRUNTIME_ROOT=/path/to/onnxruntime")
endif()
message(STATUS "  headers: ${ORT_INCLUDE}")
message(STATUS "  library: ${ORT_LIB}")

add_executable(aec_infer aec_infer.cpp wav_io.cpp)
target_include_directories(aec_infer PRIVATE ${ORT_INCLUDE})
target_link_libraries(aec_infer PRIVATE ${ORT_LIB})
if(UNIX)
  target_link_libraries(aec_infer PRIVATE pthread dl)
endif()
