include(aliasing)

find_package(libassert REQUIRED)

alias_library(deps::libassert libassert::assert)
