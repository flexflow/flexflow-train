// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/ops/transpose_attrs.struct.toml
/* proj-data
{
  "generated_from": "de62a505821a59c4b77197c100e204f7"
}
*/

#include "op-attrs/ops/transpose_attrs.dtg.h"

#include "op-attrs/dim_ordered.h"
#include "op-attrs/ff_dim.dtg.h"
#include "op-attrs/ff_dim.h"
#include <sstream>

namespace FlexFlow {
TransposeAttrs::TransposeAttrs(
    ::FlexFlow::FFOrdered<::FlexFlow::ff_dim_t> const &perm)
    : perm(perm) {}
bool TransposeAttrs::operator==(TransposeAttrs const &other) const {
  return std::tie(this->perm) == std::tie(other.perm);
}
bool TransposeAttrs::operator!=(TransposeAttrs const &other) const {
  return std::tie(this->perm) != std::tie(other.perm);
}
bool TransposeAttrs::operator<(TransposeAttrs const &other) const {
  return std::tie(this->perm) < std::tie(other.perm);
}
bool TransposeAttrs::operator>(TransposeAttrs const &other) const {
  return std::tie(this->perm) > std::tie(other.perm);
}
bool TransposeAttrs::operator<=(TransposeAttrs const &other) const {
  return std::tie(this->perm) <= std::tie(other.perm);
}
bool TransposeAttrs::operator>=(TransposeAttrs const &other) const {
  return std::tie(this->perm) >= std::tie(other.perm);
}
} // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::TransposeAttrs>::operator()(
    ::FlexFlow::TransposeAttrs const &x) const {
  size_t result = 0;
  result ^= std::hash<::FlexFlow::FFOrdered<::FlexFlow::ff_dim_t>>{}(x.perm) +
            0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}
} // namespace std

namespace nlohmann {
::FlexFlow::TransposeAttrs
    adl_serializer<::FlexFlow::TransposeAttrs>::from_json(json const &j) {
  return ::FlexFlow::TransposeAttrs{
      j.at("perm").template get<::FlexFlow::FFOrdered<::FlexFlow::ff_dim_t>>()};
}
void adl_serializer<::FlexFlow::TransposeAttrs>::to_json(
    json &j, ::FlexFlow::TransposeAttrs const &v) {
  j["__type"] = "TransposeAttrs";
  j["perm"] = v.perm;
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::TransposeAttrs>
    Arbitrary<::FlexFlow::TransposeAttrs>::arbitrary() {
  return gen::construct<::FlexFlow::TransposeAttrs>(
      gen::arbitrary<::FlexFlow::FFOrdered<::FlexFlow::ff_dim_t>>());
}
} // namespace rc

namespace FlexFlow {
std::string format_as(TransposeAttrs const &x) {
  std::ostringstream oss;
  oss << "<TransposeAttrs";
  oss << " perm=" << x.perm;
  oss << ">";
  return oss.str();
}
std::ostream &operator<<(std::ostream &s, TransposeAttrs const &x) {
  return s << fmt::to_string(x);
}
} // namespace FlexFlow