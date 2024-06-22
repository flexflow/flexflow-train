// THIS FILE WAS AUTO-GENERATED BY proj. DO NOT MODIFY IT!
// If you would like to modify this datatype, instead modify
// lib/pcg/include/pcg/parallel_computation_graph/parallel_layer_attrs.struct.toml
/* proj-data
{
  "generated_from": "1b3a0491865fd43c79afcf4939b56fae"
}
*/

#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_LAYER_ATTRS_DTG_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_LAYER_ATTRS_DTG_H

#include "fmt/format.h"
#include "nlohmann/json.hpp"
#include "op-attrs/operator_attrs.h"
#include "rapidcheck.h"
#include "utils/stack_string.h"
#include <functional>
#include <optional>
#include <ostream>
#include <tuple>

namespace FlexFlow {
struct ParallelLayerAttrs {
  ParallelLayerAttrs() = delete;
  explicit ParallelLayerAttrs(
      ::FlexFlow::PCGOperatorAttrs const &op_attrs,
      std::optional<::FlexFlow::stack_string<MAX_OPNAME>> const &name);

  bool operator==(ParallelLayerAttrs const &) const;
  bool operator!=(ParallelLayerAttrs const &) const;
  bool operator<(ParallelLayerAttrs const &) const;
  bool operator>(ParallelLayerAttrs const &) const;
  bool operator<=(ParallelLayerAttrs const &) const;
  bool operator>=(ParallelLayerAttrs const &) const;
  ::FlexFlow::PCGOperatorAttrs op_attrs;
  std::optional<::FlexFlow::stack_string<MAX_OPNAME>> name;
};
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::ParallelLayerAttrs> {
  size_t operator()(::FlexFlow::ParallelLayerAttrs const &) const;
};
} // namespace std

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::ParallelLayerAttrs> {
  static ::FlexFlow::ParallelLayerAttrs from_json(json const &);
  static void to_json(json &, ::FlexFlow::ParallelLayerAttrs const &);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::ParallelLayerAttrs> {
  static Gen<::FlexFlow::ParallelLayerAttrs> arbitrary();
};
} // namespace rc

namespace FlexFlow {
std::string format_as(ParallelLayerAttrs const &);
std::ostream &operator<<(std::ostream &, ParallelLayerAttrs const &);
} // namespace FlexFlow

#endif // _FLEXFLOW_LIB_PCG_INCLUDE_PCG_PARALLEL_COMPUTATION_GRAPH_PARALLEL_LAYER_ATTRS_DTG_H