include(aliasing)

if (FF_USE_EXTERNAL_JSON)
  find_package(nlohmann_json REQUIRED)

  alias_library(json nlohmann_json)
else()
  message(FATAL_ERROR "FF_USE_EXTERNAL_JSON is required")
endif()
