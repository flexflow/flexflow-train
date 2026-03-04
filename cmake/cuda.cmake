include(aliasing)

find_package(CUDAToolkit 11.7 REQUIRED)
enable_language(CUDA)

alias_library(cudart CUDA::cudart)
alias_library(cublas CUDA::cublas)
