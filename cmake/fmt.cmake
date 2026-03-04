include(aliasing)

if (FF_USE_EXTERNAL_FMT)
  find_package(fmt REQUIRED)
else()
  message(FATAL_ERROR "FF_USE_EXTERNAL_FMT is required")
endif()
alias_library(fmt fmt::fmt)
