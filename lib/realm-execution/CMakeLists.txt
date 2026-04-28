project(realm-execution
  LANGUAGES CXX CUDA)

file(GLOB_RECURSE SRC
     CONFIGURE_DEPENDS
     LIST_DIRECTORIES False
     src/*.cc
     src/**/*.cc
     src/cuda/*.cu
     src/**/*.cu
)

add_library(
  realm-execution
  SHARED
  ${SRC}
)

target_include_directories(
  realm-execution
  PUBLIC
    include/
  PRIVATE
    src/
)

target_link_libraries(
  realm-execution
  PUBLIC
    compiler
    kernels
    local-execution
    op-attrs
    pcg
    spdlog
    task-spec
    utils
    realm
)

define_ff_vars(realm-execution)

set_target_properties(
  realm-execution
  PROPERTIES
  CUDA_STANDARD 17
)

add_subdirectory(test)

