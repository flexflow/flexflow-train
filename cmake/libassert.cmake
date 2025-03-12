include(aliasing)

if(FF_USE_EXTERNAL_LIBASSERT)
  find_package(libassert REQUIRED)
else()
  message(FATAL_ERROR "Currently FF_USE_EXTERNAL_LIBASSERT is required")
endif()

alias_library(libassert libassert::assert)
