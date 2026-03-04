include(aliasing)

if (FF_USE_EXTERNAL_SPDLOG)
  find_package(spdlog REQUIRED)
else()
  message(FATAL_ERROR "FF_USE_EXTERNAL_SPDLOG is required")
endif()
