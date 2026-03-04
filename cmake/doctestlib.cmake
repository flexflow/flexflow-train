include(aliasing)

if (FF_USE_EXTERNAL_DOCTEST)
  find_package(doctest REQUIRED)
  include(doctest) # import doctest_discover_tests

  target_compile_definitions(
    doctest::doctest
    INTERFACE
      DOCTEST_CONFIG_REQUIRE_STRINGIFICATION_FOR_ALL_USED_TYPES
  )
  alias_library(doctest doctest::doctest)
else()
  message(FATAL_ERROR "FF_USE_EXTERNAL_DOCTEST is required")
endif()
