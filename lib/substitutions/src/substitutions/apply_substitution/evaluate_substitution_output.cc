#include "substitutions/apply_substitution/evaluate_substitution_output.h"
#include "substitutions/apply_substitution/perform_shape_inference.h"
#include "substitutions/output_graph/output_operator_attrs_assignment.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "utils/bidict/algorithms/transform_keys.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/map_values.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/permute_labelled_open_kwarg_dataflow_graph_node_ids.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_node_labels.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_value_labels.h"
#include "utils/graph/node/algorithms/generate_new_node_id_permutation.h"
#include "utils/graph/node/algorithms/new_node.dtg.h"
#include "utils/graph/open_dataflow_graph/algorithms/generate_new_input_id_permutation.h"
#include "utils/graph/open_dataflow_graph/algorithms/new_dataflow_graph_input.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/new_kwarg_dataflow_graph_input.dtg.h"

namespace FlexFlow {

std::pair<SubParallelComputationGraph, OutputExprToResultSubPCGMapping>
    evaluate_substitution_output(SubParallelComputationGraph const &spcg,
                                 Substitution const &sub,
                                 PCGPatternMatch const &match) {
  std::unordered_map<PatternNode, PCGOperatorAttrs> node_match =
      map_values(match.node_assignment.as_unordered_map(),
                 [&](parallel_layer_guid_t const &n) {
                   return get_operator_attrs(spcg, n);
                 });

  bidict<NewNode, Node> new_node_id_permutation =
      generate_new_node_id_permutation(sub.output_graph_expr.raw_graph);
  bidict<KwargDataflowGraphInput<int>, KwargDataflowGraphInput<int>> new_input_id_permutation =
       generate_bidict(get_all_kwarg_dataflow_graph_inputs(sub.output_graph_expr.raw_graph),
                       [](KwargDataflowGraphInput<int> const &i) {
                         return i;
                       });
  // bidict<NewKwargDataflowGraphInput<int>, KwargDataflowGraphInput<int>> new_input_id_permutation =
  //     generate_new_input_id_permutation(sub.output_graph_expr.raw_graph);
  // LabelledOpenKwargDataflowGraphView<OutputOperatorAttrsAssignment, std::monostate>
  //     permuted =
  //         permute_input_ids(permute_node_ids(sub.output_graph_expr.raw_graph,
  //                                            new_node_id_permutation),
  //                           new_input_id_permutation);
  // TODO(@lockshaw)(#pr): do we need the input id permutation?
  LabelledOpenKwargDataflowGraphView<OutputOperatorAttrsAssignment, std::monostate, int, TensorSlotName>
      permuted =
          permute_labelled_open_kwarg_dataflow_graph_node_ids(
              sub.output_graph_expr.raw_graph, new_node_id_permutation);

  LabelledOpenKwargDataflowGraphView<ParallelLayerAttrs, std::monostate, int, TensorSlotName>
      without_shapes = rewrite_labelled_open_kwarg_dataflow_graph_node_labels(
          permuted,
          [&](Node const &n, OutputOperatorAttrsAssignment const &attrs) {
            return ParallelLayerAttrs{
                materialize_output_operator_from_attrs_assignment(attrs,
                                                                  node_match),
                std::nullopt,
            };
          });

  bidict<input_parallel_tensor_guid_t, OutputGraphExprInput> result_input_map =
      transform_keys(transform_values(new_input_id_permutation,
                                      [](KwargDataflowGraphInput<int> const &i) {
                                        return OutputGraphExprInput{i};
                                      }),
                     [](KwargDataflowGraphInput<int> const &i) {
                       return input_parallel_tensor_guid_t{i};
                     });

  bidict<parallel_layer_guid_t, OutputGraphExprNode> result_node_map =
      transform_keys(
          transform_values(
              new_node_id_permutation,
              [](Node const &n) { return OutputGraphExprNode{n}; }),
          [](NewNode const &n) { return parallel_layer_guid_t{n.raw_node}; });

  std::unordered_map<KwargDataflowGraphInput<int>, ParallelTensorShape> input_shapes =
      map_values(map_keys(match.input_assignment,
                          [&](PatternInput const &i) {
                            return result_input_map
                                .at_r(sub.inputs_mapping.at_l(i))
                                .raw_dataflow_graph_input;
                          }),
                 [&](open_parallel_tensor_guid_t const &v) {
                   return spcg.raw_graph.at(v.raw_open_dataflow_value).shape;
                 });
  LabelledOpenKwargDataflowGraphView<ParallelLayerAttrs, ParallelTensorShape, int, TensorSlotName>
      with_shapes = perform_shape_inference(without_shapes, input_shapes);
  LabelledOpenKwargDataflowGraphView<ParallelLayerAttrs, ParallelTensorAttrs, int, TensorSlotName>
      with_attrs = rewrite_labelled_open_kwarg_dataflow_graph_value_labels(
          with_shapes,
          [](OpenKwargDataflowValue<int, TensorSlotName> const &, ParallelTensorShape const &s) {
            return ParallelTensorAttrs{
                s,
                CreateGrad::YES,
            };
          });

  return std::make_pair(SubParallelComputationGraph{with_attrs},
                        OutputExprToResultSubPCGMapping{
                            result_node_map,
                            result_input_map,
                        });
}

} // namespace FlexFlow
