include(aliasing)

if (FF_USE_EXTERNAL_NCCL)
  find_package(NCCL REQUIRED)
else()
  message(FATAL_ERROR "FF_USE_EXTERNAL_NCCL is required")
endif()

message(STATUS "NCCL_LIBRARIES = ${NCCL_LIBRARIES}")
add_library(nccl INTERFACE)
target_include_directories(nccl SYSTEM INTERFACE ${NCCL_INCLUDE_DIRS})
target_link_libraries(nccl INTERFACE ${NCCL_LIBRARIES})
