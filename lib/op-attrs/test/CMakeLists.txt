ff_add_test_executable(
  NAME
    op-attrs-tests
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    op-attrs
    kernels
    deps::doctest
    utils-test-common
)

