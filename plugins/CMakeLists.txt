cmake_minimum_required(VERSION 3.22)
project(trt_plugin LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(OUT_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)
#set(CMAKE_CXX_FLAGS "-Wno-deprecated-declarations ${CMAKE_CXX_FLAGS} -DBUILD_SYSTEM=cmake_oss")
IF (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING
        "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
ENDIF ()
add_definitions(-DPROJECT_ROOT="${PROJECT_SOURCE_DIR}")
if (${CMAKE_BUILD_TYPE} STREQUAL "Debug")
    add_definitions(-DDEBUG)
endif ()
############################################################################################
macro(set_ifndef_from_env var)
    if (NOT DEFINED ${var})
        if (DEFINED ENV{${var}})
            set(${var} $ENV{${var}})
            message(STATUS "read ${var} from environment variable")
        else ()
            message(FATAL_ERROR "variable or ${var} var is required.")
        endif ()
    endif ()
endmacro()
set_ifndef_from_env(TENSORRT_DIR)
set_ifndef_from_env(CUDNN_DIR)

find_package(CUDA ${CUDA_VERSION} REQUIRED)
find_library(CUDART_LIB cudart_static HINTS ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib lib/x64 lib64)
find_library(CUBLAS_LIB cublas REQUIRED PATH_SUFFIXES lib64 lib lib/x64 lib/stubs)
find_library(CUDNN_LIB cudnn REQUIRED
    HINTS ${CUDNN_DIR} $ENV{CUDNN_DIR}
    PATH_SUFFIXES lib64 lib/x64 lib)
find_library(TENSORRT_INFER_LIB nvinfer
    HINTS ${TENSORRT_DIR} $ENV{TENSORRT_DIR}
    PATH_SUFFIXES lib lib64 lib/x64)
find_path(TENSORRT_INCLUDE_DIR NvInfer.h
    HINTS ${TENSORRT_DIR} ${CUDA_TOOLKIT_ROOT_DIR}
    PATH_SUFFIXES include)
set(TENSORRT_LIB ${TENSORRT_INFER_LIB})
find_library(CUBLASLT_LIB cublasLt HINTS
    ${CUDA_TOOLKIT_ROOT_DIR} PATH_SUFFIXES lib64 lib lib/x64 lib/stubs)

############################################################################################
## targets
#if (DEFINED GPU_ARCHS)
#    message(STATUS "GPU_ARCHS defined as ${GPU_ARCHS}. Generating CUDA code for SM ${GPU_ARCHS}")
#    separate_arguments(GPU_ARCHS)
#else ()
#    list(APPEND GPU_ARCHS
#            53
#            60
#            61
#            70
#            75
#            )
#
#    string(REGEX MATCH "aarch64" IS_ARM "${TRT_PLATFORM_ID}")
#    if (IS_ARM)
#        # Xavier (SM72) only supported for aarch64.
#        list(APPEND GPU_ARCHS 72)
#    endif ()
#
#    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.0)
#        # Ampere GPU (SM80) support is only available in CUDA versions > 11.0
#        list(APPEND GPU_ARCHS 80)
#    endif ()
#    if (CUDA_VERSION VERSION_GREATER_EQUAL 11.1)
#        list(APPEND GPU_ARCHS 86)
#    endif ()
#
#    message(STATUS "GPU_ARCHS is not defined. Generating CUDA code for default SMs: ${GPU_ARCHS}")
#endif ()
## Generate SASS for each architecture
#foreach (arch ${GPU_ARCHS})
#    set(GENCODES "${GENCODES} -gencode arch=compute_${arch},code=sm_${arch}")
#endforeach ()
#
## Generate PTX for the last architecture in the list.
#list(GET GPU_ARCHS -1 LATEST_SM)
#set(GENCODES "${GENCODES} -gencode arch=compute_${LATEST_SM},code=compute_${LATEST_SM}")
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -Wno-deprecated-declarations")


add_subdirectory(src)
