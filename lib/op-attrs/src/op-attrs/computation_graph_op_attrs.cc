#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_op_type.h"
#include "op-attrs/ops/broadcast.h"
#include "op-attrs/ops/cast.h"
#include "op-attrs/ops/embedding.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/ops/weight.h"
#include "utils/overload.h"

namespace FlexFlow {

OperatorType get_op_type(ComputationGraphOpAttrs const &attrs) {
  return attrs.visit<OperatorType>(
      [](auto const &x) { return get_op_type(x); });
}

nlohmann::json cg_op_attrs_as_dot_json(ComputationGraphOpAttrs const &attrs) {
  nlohmann::json result = attrs;

  return result;
}

std::optional<ComputationGraphOpAttrs>
    compgraph_op_attrs_from_pcg_op_attrs(PCGOperatorAttrs const &op) {

  return op.visit<std::optional<ComputationGraphOpAttrs>>(overload{
      [&](CombineAttrs const &attrs) { return std::nullopt; },
      [&](ReductionAttrs const &attrs) { return std::nullopt; },
      [&](RepartitionAttrs const &attrs) { return std::nullopt; },
      [&](ReplicateAttrs const &attrs) { return std::nullopt; },
      [](auto const &attrs) { return ComputationGraphOpAttrs{attrs}; },
  });
}

} // namespace FlexFlow
