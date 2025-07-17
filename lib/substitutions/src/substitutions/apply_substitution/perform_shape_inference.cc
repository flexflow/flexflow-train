#include "substitutions/apply_substitution/perform_shape_inference.h"
#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/shape_inference.h"
#include "utils/containers/filtrans.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip.h"
#include "utils/graph/dataflow_graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/labelled_open_dataflow_graph/algorithms/rewrite_value_labels.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_inputs.h"

namespace FlexFlow {

LabelledOpenDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape>
    perform_shape_inference(
        LabelledOpenDataflowGraphView<ParallelLayerAttrs, std::monostate> const
            &g,
        std::unordered_map<DataflowGraphInput, ParallelTensorShape> const
            &input_shapes) {

  std::unordered_map<OpenDataflowValue, ParallelTensorShape> inferred =
      map_keys(input_shapes, [](DataflowGraphInput const &i) {
        return OpenDataflowValue{i};
      });

  for (Node const &n : get_topological_ordering(g)) {
    std::vector<ParallelTensorShape> incoming_shapes =
        transform(get_inputs(g, n),
                  [&](OpenDataflowValue const &v) { return inferred.at(v); });

    ParallelLayerAttrs n_attrs = g.at(n);

    std::vector<IncomingTensorRole> incoming_tensor_roles =
        get_incoming_tensor_roles(n_attrs.op_attrs, incoming_shapes.size());

    auto incoming_shapes_with_role =
        [&](IncomingTensorRole role) -> std::vector<ParallelTensorShape> {
      return filtrans(
          zip(incoming_shapes, incoming_tensor_roles),
          [&](std::pair<ParallelTensorShape, IncomingTensorRole> const &t)
              -> std::optional<ParallelTensorShape> {
            if (t.second == role) {
              return t.first;
            } else {
              return std::nullopt;
            }
          });
    };

    std::vector<ParallelTensorShape> input_shapes =
        incoming_shapes_with_role(IncomingTensorRole::INPUT);
    std::vector<ParallelTensorShape> weight_shapes =
        incoming_shapes_with_role(IncomingTensorRole::WEIGHT);

    std::vector<ParallelTensorShape> inferred_weight_shapes =
        get_weight_shapes(n_attrs.op_attrs, input_shapes);

    assert(weight_shapes == inferred_weight_shapes);

    std::vector<ParallelTensorShape> output_shapes =
        get_output_shapes(n_attrs.op_attrs, input_shapes);

    std::vector<DataflowOutput> outputs = get_outputs(g, n);

    for (auto const &[output, shape] : zip(outputs, output_shapes)) {
      inferred.insert({OpenDataflowValue{output}, shape});
    }
  }

  return rewrite_value_labels(
      g, [&](OpenDataflowValue const &v, std::monostate const &) {
        return inferred.at(v);
      });
}

} // namespace FlexFlow
