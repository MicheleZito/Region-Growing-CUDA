project(regionGrow)
cmake_minimum_required(VERSION 2.8)
find_package(OpenCV REQUIRED)
find_package(CUDA QUIET REQUIRED)
# Pass options to NVCC
set(
    CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -arch=sm_30
    #--relocatable-device-code true
    )
SET(CUDA_SEPARABLE_COMPILATION ON)
cuda_add_executable(regionGrow regrow_.cpp regionGrowGPU.cuh regionGrowGPU.cu)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
