// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/broadcast.struct.toml
/* proj-data
{
  "generated_from": "12715c970e8416eacbd0750f338478e5"
}
*/

#include "op-attrs/ops/broadcast.dtg.h"

#include "utils/stack_vector.h"
#include <sstream>

namespace FlexFlow {
BroadcastAttrs::BroadcastAttrs(
    ::FlexFlow::stack_vector<int, MAX_TENSOR_DIM> const &target_dims)
    : target_dims(target_dims) {}
bool BroadcastAttrs::operator==(BroadcastAttrs const &other) const {
  return std::tie(this->target_dims) == std::tie(other.target_dims);
}
bool BroadcastAttrs::operator!=(BroadcastAttrs const &other) const {
  return std::tie(this->target_dims) != std::tie(other.target_dims);
}
bool BroadcastAttrs::operator<(BroadcastAttrs const &other) const {
  return std::tie(this->target_dims) < std::tie(other.target_dims);
}
bool BroadcastAttrs::operator>(BroadcastAttrs const &other) const {
  return std::tie(this->target_dims) > std::tie(other.target_dims);
}
bool BroadcastAttrs::operator<=(BroadcastAttrs const &other) const {
  return std::tie(this->target_dims) <= std::tie(other.target_dims);
}
bool BroadcastAttrs::operator>=(BroadcastAttrs const &other) const {
  return std::tie(this->target_dims) >= std::tie(other.target_dims);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::BroadcastAttrs>::operator()(
    FlexFlow::BroadcastAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::stack_vector<int, MAX_TENSOR_DIM>>{}(
                x.target_dims) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
FlexFlow::BroadcastAttrs
    adl_serializer<FlexFlow::BroadcastAttrs>::from_json(json const &j) {
  return {j.at("target_dims")
              .template get<::FlexFlow::stack_vector<int, MAX_TENSOR_DIM>>()};
}
void adl_serializer<FlexFlow::BroadcastAttrs>::to_json(
    json &j, FlexFlow::BroadcastAttrs const &v) {
  j["__type"] = "BroadcastAttrs";
  j["target_dims"] = v.target_dims;
}
} // namespace nlohmann

namespace rc {
Gen<FlexFlow::BroadcastAttrs> Arbitrary<FlexFlow::BroadcastAttrs>::arbitrary() {
  return gen::construct<FlexFlow::BroadcastAttrs>(
      gen::arbitrary<::FlexFlow::stack_vector<int, MAX_TENSOR_DIM>>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(BroadcastAttrs const &x) {
  std::ostringstream oss;
  oss << "<BroadcastAttrs";
  oss << " target_dims=" << x.target_dims;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, BroadcastAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow