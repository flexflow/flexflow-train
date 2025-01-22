ff_add_library(
  NAME
    realm-backend
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    utils
    kernels
    local-execution
    pcg
    spdlog
    legion
)

# add_subdirectory(test)
