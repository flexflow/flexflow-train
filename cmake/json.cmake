include(aliasing)

find_package(nlohmann_json REQUIRED)

alias_library(deps::json nlohmann_json)
