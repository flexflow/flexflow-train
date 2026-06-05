ff_add_library(
  NAME
    pcg
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    utils
    deps::rapidcheck
)

add_subdirectory(ffi)
add_subdirectory(test)
