include(aliasing)

find_path(CUDNN_INCLUDE_DIR NAMES cudnn.h HINTS ${CUDAToolkit_INCLUDE_DIRS})
find_library(CUDNN_LIBRARY NAMES cudnn HINTS ${CUDAToolkit_LIBRARY_DIR})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CUDNN DEFAULT_MSG CUDNN_LIBRARY CUDNN_INCLUDE_DIR)

if(CUDNN_FOUND)
  add_library(cudnn UNKNOWN IMPORTED)
  set_target_properties(cudnn PROPERTIES
    IMPORTED_LOCATION "${CUDNN_LIBRARY}"
    INTERFACE_INCLUDE_DIRECTORIES "${CUDNN_INCLUDE_DIR}"
  )
  alias_library(deps::cudnn cudnn)
endif()
