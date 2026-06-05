include(aliasing)

find_package(Realm REQUIRED)

alias_library(deps::realm Realm::Realm)
