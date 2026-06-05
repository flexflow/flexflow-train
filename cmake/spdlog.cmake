include(aliasing)

find_package(spdlog REQUIRED)

alias_library(deps::spdlog spdlog::spdlog)
