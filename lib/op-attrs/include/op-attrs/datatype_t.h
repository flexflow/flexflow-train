// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/op-attrs/include/op-attrs/datatype_t.enum.toml

#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DATATYPE_T_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DATATYPE_T_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "rapidcheck.h"
#include <functional>
#include <ostream>
#include <string>

namespace FlexFlow {
enum class DataType { BOOL, INT32, INT64, HALF, FLOAT, DOUBLE };
std::string format_as(DataType);
std::ostream &operator<<(std::ostream &, DataType);
void to_json(::nlohmann::json &, DataType);
void from_json(::nlohmann::json const &, DataType &);
} // namespace FlexFlow
namespace std {
template <>
struct hash<FlexFlow::DataType> {
  size_t operator()(FlexFlow::DataType) const;
};
} // namespace std
namespace rc {
template <>
struct Arbitrary<FlexFlow::DataType> {
  static Gen<FlexFlow::DataType> arbitrary();
};
} // namespace rc

#endif // _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DATATYPE_T_H