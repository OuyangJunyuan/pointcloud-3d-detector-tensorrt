cmake_minimum_required(VERSION 3.22)
project(point_detection LANGUAGES CXX)
include(cmake/utils.cmake)
add_subdirectory(plugins)
########################################################################################################################
# setup
########################################################################################################################
set(CMAKE_CXX_STANDARD 17)

set_ifndef_from_env(CUDNN_DIR)
set_ifndef_from_env(TENSORRT_DIR)
set_ifndef_from_default(TRT_QUANTIZE FP16)
set_ifndef_from_default(CMAKE_BUILD_TYPE Release)
########################################################################################################################
# Dependencies
########################################################################################################################
find_package(catkin REQUIRED COMPONENTS roscpp sensor_msgs visualization_msgs pcl_conversions QUIET)
find_package(PCL REQUIRED COMPONENTS filters QUIET)
find_package(yaml-cpp REQUIRED QUIET)
find_package(Boost REQUIRED COMPONENTS filesystem QUIET)
find_package(CUDA ${CUDA_VERSION} REQUIRED QUIET)

find_library(TENSORRT_INFER_LIB nvinfer
    HINTS ${TENSORRT_DIR} $ENV{TENSORRT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_library(TENSORRT_ONNX_PARSER_LIB nvonnxparser
    HINTS ${TENSORRT_DIR} $ENV{TENSORRT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)

set(TENSORRT_LIB ${TENSORRT_INFER_LIB} ${TENSORRT_ONNX_PARSER_LIB})
find_library(CUDNN_LIB cudnn REQUIRED
    HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR}
    PATH_SUFFIXES lib64 lib/x64 lib)
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/x64 lib/stubs)
########################################################################################################################
## build
########################################################################################################################
catkin_package(
    #  INCLUDE_DIRS include
    #  LIBRARIES point_detection
    #  CATKIN_DEPENDS other_catkin_pkg
    #  DEPENDS system_lib
)
add_definitions(-DTRT_INT8=0)
add_definitions(-DTRT_FP16=1)
add_definitions(-DTRT_FP32=2)
add_definitions(-DTRT_QUANTIZE=TRT_${TRT_QUANTIZE})
add_definitions(-DPROJECT_ROOT="${PROJECT_SOURCE_DIR}")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif ()

set(INC_DIRS
    ./
    ${TENSORRT_INCLUDE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${catkin_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS})

set(LINK_DIRS
    ${CUDNN_DIR}/lib)

set(LINK_LIBS
    yaml-cpp
    ${TENSORRT_LIB}
    ${CUDNN_LIB}
    ${CUDA_LIBRARIES}
    ${CUBLASLT_LIB}
    ${Boost_LIBRARIES}
    ${catkin_LIBRARIES}
    ${PCL_LIBRARIES})
##############
## targets
##############
add_executable(point_detector src/point_detector.cpp)
target_include_directories(point_detector PRIVATE ${INC_DIRS})
target_link_directories(point_detector PRIVATE ${LINK_DIRS})
target_link_libraries(point_detector PRIVATE ${LINK_LIBS})


message(STATUS "******** Summary ********")
message(STATUS "  CMake version             : ${CMAKE_VERSION}")
message(STATUS "  CMake command             : ${CMAKE_COMMAND}")
message(STATUS "  System                    : ${CMAKE_SYSTEM_NAME}")
message(STATUS "  C++ compiler              : ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ compiler version      : ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  Build type                : ${CMAKE_BUILD_TYPE}")
message(STATUS "  CXX flags                 : ${CMAKE_CXX_FLAGS}")
message(STATUS "  CMAKE_PREFIX_PATH         : ${CMAKE_PREFIX_PATH}")
message(STATUS "  CMAKE_INSTALL_PREFIX      : ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  CMAKE_MODULE_PATH         : ${CMAKE_MODULE_PATH}")
message(STATUS "")
message(STATUS "  CUDA_VERSION              : ${CUDA_VERSION}")
message(STATUS "  CUDA_TOOLKIT_ROOT_DIR     : ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "  CUDA_LIBRARIES            : ${CUDA_LIBRARIES}")
message(STATUS "  CUDA_INCLUDE_DIRS:        : ${CUDA_INCLUDE_DIRS}")
message(STATUS "  CUDART_LIB                : ${CUDART_LIB}")
message(STATUS "  CUBLAS_LIB                : ${CUBLAS_LIB}")
message(STATUS "  TENSORRT_LIB              : ${TENSORRT_LIB}")
message(STATUS "  TENSORRT_INCLUDE_DIR      : ${TENSORRT_INCLUDE_DIR}")
message(STATUS "  CUDNN_LIB                 : ${CUDNN_LIB}")
message(STATUS "  CUBLASLT_LIB              : ${CUBLASLT_LIB}")
