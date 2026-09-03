ff_add_test_executable(
  NAME
    utils-tests
  SRC_PATTERNS
    src/utils/*.cc
  PRIVATE_INCLUDE
    src/
  DEPS
    utils
    deps::doctest
    utils-test-common
    deps::realm
)

add_subdirectory(common)
