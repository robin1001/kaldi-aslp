cmake_minimum_required(VERSION 2.8)

project(ctc_release)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O2")

include_directories(include)

FIND_PACKAGE(CUDA 6.5)
MESSAGE(STATUS "cuda found ${CUDA_FOUND}")

# need to be at least 30 or __shfl_down in reduce wont compile
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_30,code=sm_30 -O2")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_35,code=sm_35")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_50,code=sm_50")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=sm_52")

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --std=c++11")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fopenmp")

#always skip for linux
set(CMAKE_SKIP_RPATH TRUE)

IF (CUDA_FOUND)

    MESSAGE(STATUS "Building shared library with GPU support")

    #CUDA_ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cu src/reduce.cu)
    CUDA_ADD_LIBRARY(warpctc STATIC src/ctc_entrypoint.cu src/reduce.cu)
    TARGET_LINK_LIBRARIES(warpctc ${CUDA_curand_LIBRARY})

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    cuda_add_executable(test_gpu tests/test_gpu.cu)
    TARGET_LINK_LIBRARIES(test_gpu warpctc ${CUDA_curand_LIBRARY})

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "${CMAKE_BINARY_DIR}"
            LIBRARY DESTINATION "${CMAKE_BINARY_DIR}"
            ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}")

ELSE()
    MESSAGE(STATUS "Building shared library with no GPU support")

    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O2")

    #ADD_LIBRARY(warpctc SHARED src/ctc_entrypoint.cpp)
    ADD_LIBRARY(warpctc STATIC src/ctc_entrypoint.cpp)

    add_executable(test_cpu tests/test_cpu.cpp )
    TARGET_LINK_LIBRARIES(test_cpu warpctc)
    SET_TARGET_PROPERTIES(test_cpu PROPERTIES COMPILE_FLAGS "${CMAKE_CXX_FLAGS} --std=c++11")

    INSTALL(TARGETS warpctc
            RUNTIME DESTINATION "${CMAKE_BINARY_DIR}"
            LIBRARY DESTINATION "${CMAKE_BINARY_DIR}"
            ARCHIVE DESTINATION "${CMAKE_BINARY_DIR}")
ENDIF()


