// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/initializers/constant_initializer_attrs.struct.toml
/* proj-data
{
  "generated_from": "0162b9c49fe6cbfc65410c6fa8dec427"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_INITIALIZERS_CONSTANT_INITIALIZER_ATTRS_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_INITIALIZERS_CONSTANT_INITIALIZER_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/datatype.h"
#include "utils/json.h"
#include <functional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct ConstantInitializerAttrs {
  ConstantInitializerAttrs() = delete;
  ConstantInitializerAttrs(::FlexFlow::DataTypeValue const &value);

  bool operator==(ConstantInitializerAttrs const &) const;
  bool operator!=(ConstantInitializerAttrs const &) const;
  bool operator<(ConstantInitializerAttrs const &) const;
  bool operator>(ConstantInitializerAttrs const &) const;
  bool operator<=(ConstantInitializerAttrs const &) const;
  bool operator>=(ConstantInitializerAttrs const &) const;
  ::FlexFlow::DataTypeValue value;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<FlexFlow::ConstantInitializerAttrs> {
  size_t operator()(FlexFlow::ConstantInitializerAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<FlexFlow::ConstantInitializerAttrs> {
  static FlexFlow::ConstantInitializerAttrs from_json(json const &);
  static void to_json(json &, FlexFlow::ConstantInitializerAttrs const &);
};
} // namespace nlohmann

namespace FlexFlow {
std::string format_as(ConstantInitializerAttrs const &);
std::ostream &operator<<(std::ostream &, ConstantInitializerAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_INITIALIZERS_CONSTANT_INITIALIZER_ATTRS_DTG_H