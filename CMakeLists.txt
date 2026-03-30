cmake_minimum_required(VERSION 3.20)

project(wGInfer LANGUAGES CXX)

option(WGINFER_USE_OPENBLAS "Use OpenBLAS for CPU linear kernels" OFF)
option(WGINFER_ENABLE_NVIDIA "Build NVIDIA CUDA backend" ON)
option(WGINFER_ENABLE_METAX "Build MetaX backend" OFF)
option(WGINFER_COPY_TO_PYTHON "Copy built shared library into python package" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(WGINFER_ENABLE_NVIDIA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set(CMAKE_CUDA_ARCHITECTURES 75 CACHE STRING "CUDA architectures to build" FORCE)
    endif()
endif()

function(wginfer_target_defaults target_name)
    target_include_directories(
        ${target_name}
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    if(MSVC)
        target_compile_options(
            ${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:/W4>
                $<$<COMPILE_LANGUAGE:CXX>:/WX>
        )
    else()
        target_compile_options(
            ${target_name}
            PRIVATE
                $<$<COMPILE_LANGUAGE:CXX>:-Wall>
                $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
                $<$<COMPILE_LANGUAGE:CXX>:-Werror>
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-unknown-pragmas>
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-unused-parameter>
                $<$<COMPILE_LANGUAGE:CXX>:-Wno-missing-field-initializers>
        )
    endif()
endfunction()

function(wginfer_append_unique_if_exists list_var path_value)
    if(path_value AND EXISTS "${path_value}")
        set(local_list "${${list_var}}")
        list(FIND local_list "${path_value}" existing_index)
        if(existing_index EQUAL -1)
            list(APPEND local_list "${path_value}")
        endif()
        set(${list_var} "${local_list}" PARENT_SCOPE)
    endif()
endfunction()

file(GLOB WGINFER_UTIL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/utils/*.cpp")
file(GLOB WGINFER_DEVICE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/device/*.cpp")
file(GLOB WGINFER_DEVICE_CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/device/cpu/*.cpp")
file(GLOB WGINFER_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/core/*/*.cpp")
file(GLOB WGINFER_TENSOR_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/tensor/*.cpp")
file(GLOB WGINFER_OP_FRONTEND_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*/*.cpp")
file(GLOB WGINFER_OP_CPU_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*/cpu/*.cpp")
file(GLOB WGINFER_MODEL_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/models/*/*.cpp")

set(WGINFER_SOURCES
    ${WGINFER_UTIL_SOURCES}
    ${WGINFER_DEVICE_SOURCES}
    ${WGINFER_DEVICE_CPU_SOURCES}
    ${WGINFER_CORE_SOURCES}
    ${WGINFER_TENSOR_SOURCES}
    ${WGINFER_OP_FRONTEND_SOURCES}
    ${WGINFER_OP_CPU_SOURCES}
    ${WGINFER_MODEL_SOURCES}
)

if(WGINFER_ENABLE_NVIDIA)
    file(GLOB WGINFER_DEVICE_NVIDIA_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/device/nvidia/*.cu")
    file(GLOB WGINFER_OP_NVIDIA_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*/nvidia/*.cu")
    list(APPEND WGINFER_SOURCES
        ${WGINFER_DEVICE_NVIDIA_SOURCES}
        ${WGINFER_OP_NVIDIA_SOURCES}
    )
endif()

add_library(wginfer SHARED ${WGINFER_SOURCES})
add_library(wginfer::wginfer ALIAS wginfer)

wginfer_target_defaults(wginfer)

set_target_properties(
    wginfer
    PROPERTIES
        OUTPUT_NAME wginfer
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/lib"
        ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin"
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(wginfer PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found, building without OpenMP acceleration.")
endif()

if(WGINFER_USE_OPENBLAS)
    find_package(OpenBLAS REQUIRED)
    target_compile_definitions(wginfer PRIVATE WGINFER_USE_OPENBLAS)
    if(OpenBLAS_INCLUDE_DIRS)
        target_include_directories(wginfer PRIVATE ${OpenBLAS_INCLUDE_DIRS})
    endif()
    target_link_libraries(wginfer PRIVATE ${OpenBLAS_LIBRARIES})

    foreach(extra_dir IN ITEMS /usr/include/x86_64-linux-gnu /usr/include)
        if(EXISTS "${extra_dir}")
            target_include_directories(wginfer PRIVATE "${extra_dir}")
        endif()
    endforeach()
endif()

if(WGINFER_ENABLE_NVIDIA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(wginfer PRIVATE ENABLE_NVIDIA_API)
    target_link_libraries(wginfer PRIVATE CUDA::cudart CUDA::cublas)
    set_target_properties(
        wginfer
        PROPERTIES
            CUDA_SEPARABLE_COMPILATION ON
            CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    if(NOT MSVC)
        target_compile_options(
            wginfer
            PRIVATE
                $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-fPIC,-Wno-unknown-pragmas>
        )
    endif()
endif()

if(WGINFER_ENABLE_METAX)
    message(STATUS "MetaX CMake path is enabled but not validated in this environment.")
    target_compile_definitions(wginfer PRIVATE ENABLE_METAX_API)

    set(WGINFER_METAX_ROOT_CANDIDATES
        "$ENV{MACA_HOME}"
        "/opt/maca"
        "/usr/local/maca"
        "/opt/maca-3.3.0"
        "/opt/maca-3.2.0"
        "/opt/maca-3.1.0"
    )

    set(WGINFER_METAX_INCLUDE_DIRS)
    set(WGINFER_METAX_LINK_DIRS)
    foreach(root_dir IN LISTS WGINFER_METAX_ROOT_CANDIDATES)
        if(root_dir)
            wginfer_append_unique_if_exists(WGINFER_METAX_INCLUDE_DIRS "${root_dir}/include")
            wginfer_append_unique_if_exists(WGINFER_METAX_INCLUDE_DIRS "${root_dir}/include/mcr")
            wginfer_append_unique_if_exists(WGINFER_METAX_INCLUDE_DIRS "${root_dir}/mxgpu_llvm/include")

            wginfer_append_unique_if_exists(WGINFER_METAX_LINK_DIRS "${root_dir}/lib")
            wginfer_append_unique_if_exists(WGINFER_METAX_LINK_DIRS "${root_dir}/lib64")
            wginfer_append_unique_if_exists(WGINFER_METAX_LINK_DIRS "${root_dir}/mxgpu_llvm/lib")
            wginfer_append_unique_if_exists(WGINFER_METAX_LINK_DIRS "${root_dir}/mxgpu_llvm/lib64")
        endif()
    endforeach()

    set(WGINFER_MXCC "$ENV{MXCC}")
    if(NOT WGINFER_MXCC AND DEFINED ENV{MACA_HOME})
        set(candidate_mxcc "$ENV{MACA_HOME}/mxgpu_llvm/bin/mxcc")
        if(EXISTS "${candidate_mxcc}")
            set(WGINFER_MXCC "${candidate_mxcc}")
        endif()
    endif()
    if(NOT WGINFER_MXCC)
        find_program(WGINFER_MXCC NAMES mxcc)
    endif()
    if(NOT WGINFER_MXCC)
        message(FATAL_ERROR "WGINFER_ENABLE_METAX=ON but mxcc was not found. Set MXCC or MACA_HOME first.")
    endif()

    if(WGINFER_METAX_INCLUDE_DIRS)
        target_include_directories(wginfer PRIVATE ${WGINFER_METAX_INCLUDE_DIRS})
    endif()
    if(WGINFER_METAX_LINK_DIRS)
        target_link_directories(wginfer PRIVATE ${WGINFER_METAX_LINK_DIRS})
    endif()
    target_link_libraries(wginfer PRIVATE mcruntime mxc-runtime64 runtime_cu mcblas)

    set(WGINFER_METAX_GEN_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/metax")
    file(MAKE_DIRECTORY "${WGINFER_METAX_GEN_DIR}")

    set(WGINFER_METAX_DEVICE_SOURCES
        "${CMAKE_CURRENT_SOURCE_DIR}/src/device/metax/metax_resource.maca"
        "${CMAKE_CURRENT_SOURCE_DIR}/src/device/metax/metax_runtime_api.maca"
    )
    set(WGINFER_METAX_DEVICE_WRAPPERS)
    foreach(metax_source IN LISTS WGINFER_METAX_DEVICE_SOURCES)
        get_filename_component(wrapper_base "${metax_source}" NAME)
        set(wrapper_file "${WGINFER_METAX_GEN_DIR}/${wrapper_base}_wrapper.cpp")
        file(WRITE "${wrapper_file}" "#include \"${metax_source}\"\n")
        list(APPEND WGINFER_METAX_DEVICE_WRAPPERS "${wrapper_file}")
    endforeach()
    target_sources(wginfer PRIVATE ${WGINFER_METAX_DEVICE_WRAPPERS})

    file(GLOB WGINFER_METAX_OP_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*/metax/*.maca")

    if(WGINFER_METAX_OP_SOURCES)
        set(WGINFER_METAX_OP_OBJ_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated/metax_ops_obj")
        file(MAKE_DIRECTORY "${WGINFER_METAX_OP_OBJ_DIR}")

        set(WGINFER_METAX_INCLUDE_ARGS
            "-I${CMAKE_CURRENT_SOURCE_DIR}/include"
            "-I${CMAKE_CURRENT_SOURCE_DIR}/src"
        )
        foreach(include_dir IN LISTS WGINFER_METAX_INCLUDE_DIRS)
            list(APPEND WGINFER_METAX_INCLUDE_ARGS "-I${include_dir}")
        endforeach()

        set(WGINFER_METAX_OP_OBJECTS)
        foreach(metax_source IN LISTS WGINFER_METAX_OP_SOURCES)
            get_filename_component(metax_dir "${metax_source}" DIRECTORY)
            get_filename_component(op_dir "${metax_dir}" DIRECTORY)
            get_filename_component(op_name "${op_dir}" NAME)
            get_filename_component(source_name "${metax_source}" NAME)
            string(REPLACE "." "_" source_stub "${source_name}")
            set(object_file "${WGINFER_METAX_OP_OBJ_DIR}/${op_name}_${source_stub}.o")

            add_custom_command(
                OUTPUT "${object_file}"
                COMMAND "${WGINFER_MXCC}"
                        -std=c++17
                        -O3
                        -fPIC
                        -Wno-unknown-pragmas
                        -DENABLE_METAX_API
                        ${WGINFER_METAX_INCLUDE_ARGS}
                        -c "${metax_source}"
                        -o "${object_file}"
                DEPENDS "${metax_source}"
                VERBATIM
            )
            list(APPEND WGINFER_METAX_OP_OBJECTS "${object_file}")
        endforeach()

        set_source_files_properties(
            ${WGINFER_METAX_OP_OBJECTS}
            PROPERTIES
                EXTERNAL_OBJECT TRUE
                GENERATED TRUE
        )
        target_sources(wginfer PRIVATE ${WGINFER_METAX_OP_OBJECTS})
    endif()
endif()

if(WGINFER_COPY_TO_PYTHON)
    add_custom_command(
        TARGET wginfer
        POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E make_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/python/wginfer/libwginfer"
        COMMAND "${CMAKE_COMMAND}" -E copy_if_different
                "$<TARGET_FILE:wginfer>"
                "${CMAKE_CURRENT_SOURCE_DIR}/python/wginfer/libwginfer/$<TARGET_FILE_NAME:wginfer>"
        VERBATIM
    )
endif()

install(
    TARGETS wginfer
    LIBRARY DESTINATION .
    RUNTIME DESTINATION .
    ARCHIVE DESTINATION .
)
