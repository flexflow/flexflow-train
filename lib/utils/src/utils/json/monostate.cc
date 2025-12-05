#include "utils/json/monostate.h"
#include <libassert/assert.hpp>

namespace nlohmann {

void adl_serializer<std::monostate>::to_json(json &j, std::monostate) {
  j = nullptr;
}

void adl_serializer<std::monostate>::from_json(json const &j, std::monostate &x) {
  ASSERT(j == nullptr);
  x = std::monostate{};
}


} // namespace nlohmann
