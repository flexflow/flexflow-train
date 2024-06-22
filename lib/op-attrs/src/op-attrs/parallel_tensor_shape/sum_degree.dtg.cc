// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/parallel_tensor_shape/sum_degree.struct.toml
/* proj-data
{
  "generated_from": "e94a05618f2ad92dd7b3328a1d9c6786"
}
*/

#include "op-attrs/parallel_tensor_shape/sum_degree.dtg.h"

#include <sstream>

namespace FlexFlow {
SumDegree::SumDegree(int const &value) : value(value) {}
bool SumDegree::operator==(SumDegree const &other) const {
  return std::tie(this->value) == std::tie(other.value);
}
bool SumDegree::operator!=(SumDegree const &other) const {
  return std::tie(this->value) != std::tie(other.value);
}
bool SumDegree::operator<(SumDegree const &other) const {
  return std::tie(this->value) < std::tie(other.value);
}
bool SumDegree::operator>(SumDegree const &other) const {
  return std::tie(this->value) > std::tie(other.value);
}
bool SumDegree::operator<=(SumDegree const &other) const {
  return std::tie(this->value) <= std::tie(other.value);
}
bool SumDegree::operator>=(SumDegree const &other) const {
  return std::tie(this->value) >= std::tie(other.value);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::SumDegree>::operator()(
    ::FlexFlow::SumDegree const &x) const {
  size_t result = 0;
  result ^=
      std::hash<int>{}(x.value) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::SumDegree
    adl_serializer<::FlexFlow::SumDegree>::from_json(json const &j) {
  return ::FlexFlow::SumDegree{j.at("value").template get<int>()};
}
void adl_serializer<::FlexFlow::SumDegree>::to_json(
    json &j, ::FlexFlow::SumDegree const &v) {
  j["__type"] = "SumDegree";
  j["value"] = v.value;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::SumDegree> Arbitrary<::FlexFlow::SumDegree>::arbitrary() {
  return gen::construct<::FlexFlow::SumDegree>(gen::arbitrary<int>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(SumDegree const &x) {
  std::ostringstream oss;
  oss << "<SumDegree";
  oss << " value=" << x.value;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, SumDegree const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow