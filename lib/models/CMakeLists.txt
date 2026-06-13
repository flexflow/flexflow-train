ff_add_library(
  NAME
    models
  SRC_PATTERNS
    src/*.cc
  PUBLIC_INCLUDE
    include/
  PRIVATE_INCLUDE
    src/
  DEPS
    op-attrs
    utils
    pcg
    deps::rapidcheck
)

add_subdirectory(test)
