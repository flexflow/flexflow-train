find_package(NCCL REQUIRED)

add_library(nccl INTERFACE)
target_include_directories(nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
target_link_libraries(nccl INTERFACE ${NCCL_LIBRARIES})
