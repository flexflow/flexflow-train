ff_add_test_executable(
  NAME
    task-spec-tests
  SRC_PATTERNS
    src/task-spec/dynamic_graph/*.cc
  PRIVATE_INCLUDE 
    src/
  DEPS
    doctest
    utils-test-common
    # local-execution
    kernels
    task-spec
    op-attrs
)
