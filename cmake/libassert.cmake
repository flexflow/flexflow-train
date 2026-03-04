include(aliasing)

if(FF_USE_EXTERNAL_LIBASSERT)
  find_package(libassert REQUIRED)
else()
  message(FATAL_ERROR "FF_USE_EXTERNAL_LIBASSERT is required")
endif()

alias_library(libassert libassert::assert)
