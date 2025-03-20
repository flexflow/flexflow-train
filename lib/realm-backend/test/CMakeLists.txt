ff_add_executable(
  NAME
    realm-backend-tests
  SRC_PATTERNS
    src/*.cc
  PRIVATE_INCLUDE 
    src/
  DEPS
    realm-backend
)

set(FF_TEST_EXEC_NAME "realm-backend-tests")
add_custom_command(
  TARGET ${FF_TEST_EXEC_NAME} POST_BUILD
  COMMAND ${CMAKE_COMMAND} -DFF_TEST_EXEC_NAME=${FF_TEST_EXEC_NAME} -P ${CMAKE_CURRENT_LIST_DIR}/modify_test_commands.cmake
  DEPENDS ${FF_TEST_EXEC_NAME}
)
