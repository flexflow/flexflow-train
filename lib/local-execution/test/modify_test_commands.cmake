# modify_test_commands.cmake

file(GLOB ctest_tests_files "${CMAKE_CURRENT_BINARY_DIR}/${FF_TEST_EXEC_NAME}_tests-*.cmake")

foreach(ctest_tests_file IN LISTS ctest_tests_files)
  file(READ "${ctest_tests_file}" content)

  # add nix run prefix
  string(REGEX REPLACE 
    "add_test\\([ \t\r\n]*\\[==\\[([^]]+)\\]==\\][ \t\r\n]+([^ ]+)[ \t\r\n]+\\[==\\[([^]]+)\\]==\\]\\)" 
    "add_test( [==[\\1]==] nixGL -- \\2 [==[\\3]==])" 
    content "${content}")

  # add environment
  # string(REGEX REPLACE 
  #   "set_tests_properties\\([ \t\r\n]*\\[==\\[([^]]+)\\]==\\][ \t\r\n]+PROPERTIES[ \t\r\n]+([^)]+)\\)" 
  #   "set_tests_properties( [==[\\1]==] PROPERTIES \\2 ENVIRONMENT \"NIXPKGS_ALLOW_UNFREE=1\")" 
  #   content "${content}")

  file(WRITE "${ctest_tests_file}" "${content}")
endforeach()
