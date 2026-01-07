#include "substitutions/apply_substitution/output_expr_to_result_sub_pcg_mapping.h"
#include "substitutions/output_graph/output_graph_expr.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/bidict/algorithms/merge_disjoint_bidicts.h"
#include "utils/containers/values.h"
#include "utils/containers/zip_values_strict.h"

namespace FlexFlow {

bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput>
    get_output_graph_expr_output_mapping(
        OutputExprToResultSubPCGMapping const &m,
        OutputGraphExpr const &output_graph_expr,
        SubParallelComputationGraph const &spcg) {
  bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput> result;

  for (auto const &[parallel_layer, output_graph_expr_node] : m.node_mapping) {
    std::unordered_map<TensorSlotName, parallel_tensor_guid_t> layer_outputs =
        get_layer_outputs(spcg, parallel_layer);
    std::unordered_map<TensorSlotName, OutputGraphExprNodeOutput>
        output_graph_expr_outputs =
            get_node_outputs(output_graph_expr, output_graph_expr_node);

    bidict<parallel_tensor_guid_t, OutputGraphExprNodeOutput>
        mapping_for_layer = bidict_from_pairs(values(
            zip_values_strict(layer_outputs, output_graph_expr_outputs)));

    result = merge_disjoint_bidicts(result, mapping_for_layer);
  }

  return result;
}

} // namespace FlexFlow
