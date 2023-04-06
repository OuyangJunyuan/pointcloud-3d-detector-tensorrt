set(TARGET_NAME rd3d_trt_plugin)
set(SHARED_TARGET ${TARGET_NAME})
set(PLUGIN_SOURCES)
set(PLUGIN_CU_SOURCES)
set(PLUGIN_LISTS
    havSamplingPlugin
    ballQueryPlugin
    FPSamplingPlugin
    nms3DPlugin)

add_subdirectory(common)
foreach (PLUGIN_ITER ${PLUGIN_LISTS})
    include_directories(${PLUGIN_ITER})
    add_subdirectory(${PLUGIN_ITER})
endforeach (PLUGIN_ITER)

set_source_files_properties(${PLUGIN_CU_SOURCES} PROPERTIES COMPILE_FLAGS "${GENCODES} ${ENABLED_SMS}")
list(APPEND PLUGIN_SOURCES "${PLUGIN_CU_SOURCES}")

add_library(${SHARED_TARGET} SHARED
    ${PLUGIN_SOURCES}
    )

target_include_directories(${SHARED_TARGET}
    PUBLIC ${TENSORRT_INCLUDE_DIR}
    PUBLIC ${CUDA_INCLUDE_DIRS}
    PRIVATE ${PROJECT_SOURCE_DIR}/common
    PRIVATE ./
    )

if (CUDA_VERSION VERSION_LESS 11.0)
    target_include_directories(${SHARED_TARGET} PUBLIC ${CUB_ROOT_DIR})
endif ()
set_target_properties(${SHARED_TARGET} PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY "${OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY "${OUT_DIR}"
    )

set(PLUGIN_EXPORT_MAP ${CMAKE_CURRENT_SOURCE_DIR}/exports.map)
set_target_properties(${SHARED_TARGET} PROPERTIES LINK_FLAGS "-Wl,--exclude-libs,ALL -Wl,--version-script=${PLUGIN_EXPORT_MAP} -Wl,--no-undefined")

target_link_libraries(${SHARED_TARGET}
    ${CUBLAS_LIB}
    ${CUDNN_LIB}
    ${TENSORRT_LIB}
    ${CUDART_LIB}
    ${CUBLASLT_LIB}
    )
