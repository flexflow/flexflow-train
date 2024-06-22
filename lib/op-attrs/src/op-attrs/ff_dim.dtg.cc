// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ff_dim.struct.toml
/* proj-data
{
  "generated_from": "a5fa89a024e95c4f2d52681a74cab30f"
}
*/

#include "op-attrs/ff_dim.dtg.h"

#include <sstream>

namespace FlexFlow {
ff_dim_t::ff_dim_t(int const &value) : value(value) {}
bool ff_dim_t::operator==(ff_dim_t const &other) const {
  return std::tie(this->value) == std::tie(other.value);
}
bool ff_dim_t::operator!=(ff_dim_t const &other) const {
  return std::tie(this->value) != std::tie(other.value);
}
bool ff_dim_t::operator<(ff_dim_t const &other) const {
  return std::tie(this->value) < std::tie(other.value);
}
bool ff_dim_t::operator>(ff_dim_t const &other) const {
  return std::tie(this->value) > std::tie(other.value);
}
bool ff_dim_t::operator<=(ff_dim_t const &other) const {
  return std::tie(this->value) <= std::tie(other.value);
}
bool ff_dim_t::operator>=(ff_dim_t const &other) const {
  return std::tie(this->value) >= std::tie(other.value);
}
} // namespace FlexFlow

namespace std {
size_t
    hash<FlexFlow::ff_dim_t>::operator()(::FlexFlow::ff_dim_t const &x) const {
  size_t result = 0;
  result ^=
      std::hash<int>{}(x.value) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::ff_dim_t
    adl_serializer<::FlexFlow::ff_dim_t>::from_json(json const &j) {
  return ::FlexFlow::ff_dim_t{j.at("value").template get<int>()};
}
void adl_serializer<::FlexFlow::ff_dim_t>::to_json(
    json &j, ::FlexFlow::ff_dim_t const &v) {
  j["__type"] = "ff_dim_t";
  j["value"] = v.value;
}
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(ff_dim_t const &x) {
  std::ostringstream oss;
  oss << "<ff_dim_t";
  oss << " value=" << x.value;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, ff_dim_t const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow