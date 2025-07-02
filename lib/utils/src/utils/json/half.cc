#include "utils/json/half.h"

namespace nlohmann {

void adl_serializer<half>::to_json(json &j, half x) {
  j = static_cast<float>(x);
}

void adl_serializer<half>::from_json(json const &j, half &x) {
  x = j.get<float>();
}

}
