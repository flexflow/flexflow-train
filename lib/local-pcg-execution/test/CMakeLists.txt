ff_add_test_executable(
  NAME
    local-pcg-execution-tests
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE 
    src/
  DEPS
    doctest
    utils-test-common
    local-pcg-execution
    kernels
    op-attrs
    task-spec
)

