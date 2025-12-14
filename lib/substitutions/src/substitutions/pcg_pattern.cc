#include "substitutions/pcg_pattern.h"
#include "substitutions/operator_pattern/satisfies_pattern.h"
#include "substitutions/pcg_pattern_match.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/tensor_pattern/satisfies_pattern.h"
#include "substitutions/unlabelled/find_pattern_matches.h"
#include "substitutions/unlabelled/pattern_value.h"
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/containers/map_values.h"
#include "utils/containers/transform.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_outputs_for_node.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_graph_inputs.h"

namespace FlexFlow {

std::unordered_set<PatternNode> get_nodes(PCGPattern const &p) {
  std::unordered_set<Node> raw_nodes = get_nodes(p.raw_graph);

  return transform(raw_nodes, [](Node const &n) { return PatternNode{n}; });
}

static MatchAdditionalCriterion
    pcg_pattern_criteria(PCGPattern const &pattern,
                         SubParallelComputationGraph const &pcg) {
  return MatchAdditionalCriterion{
      [&](PatternNode const &patternNode, Node const &pcgNode) {
        return operator_satisfies_pattern(
            get_operator_attrs(pcg, parallel_layer_guid_t{pcgNode}),
            get_operator_pattern(pattern, patternNode));
      },
      [&](PatternValue const &patternValue, OpenKwargDataflowValue<int, TensorSlotName> const &pcgValue) {
        return parallel_tensor_satisfies_pattern(
            get_parallel_tensor_attrs(pcg,
                                      open_parallel_tensor_guid_t{pcgValue}),
            get_tensor_pattern(pattern, patternValue));
      }};
}

std::vector<PCGPatternMatch>
    find_pattern_matches(PCGPattern const &pattern,
                         SubParallelComputationGraph const &pcg) {
  std::vector<UnlabelledKwargDataflowGraphPatternMatch> unlabelled_matches =
      find_unlabelled_pattern_matches(
        get_unlabelled_pattern(pattern),
        pcg.raw_graph,
        pcg_pattern_criteria(pattern, pcg));
  auto pcg_match_from_unlabelled_match =
      [](UnlabelledKwargDataflowGraphPatternMatch const &m) {
        return PCGPatternMatch{
            transform_values(
                m.node_assignment,
                [](Node const &n) { return parallel_layer_guid_t{n}; }),
            map_values(m.input_assignment,
                       [](OpenKwargDataflowValue<int, TensorSlotName> const &i) 
                       {
                         return open_parallel_tensor_guid_t{i};
                       }),
        };
      };

  return transform(unlabelled_matches, pcg_match_from_unlabelled_match);
}

UnlabelledGraphPattern get_unlabelled_pattern(PCGPattern const &p) {
  return UnlabelledGraphPattern{p.raw_graph};
}

TensorAttributePattern get_tensor_pattern(PCGPattern const &p,
                                          PatternValue const &v) {
  return p.raw_graph.at(raw_open_dataflow_value_from_pattern_value(v));
}

OperatorAttributePattern get_operator_pattern(PCGPattern const &p,
                                              PatternNode const &n) {
  return p.raw_graph.at(n.raw_node);
}

std::unordered_set<PatternInput> get_inputs(PCGPattern const &p) {
  std::unordered_set<KwargDataflowGraphInput<int>> raw_inputs =
      get_all_kwarg_dataflow_graph_inputs(p.raw_graph);

  return transform(raw_inputs,
                   [](KwargDataflowGraphInput<int> const &i) { return PatternInput{i}; });
}

std::unordered_map<TensorSlotName, PatternNodeOutput>
    get_pattern_node_outputs(PCGPattern const &pattern,
                             PatternNode const &node) {
  std::unordered_map<TensorSlotName, KwargDataflowOutput<TensorSlotName>> raw_outputs =
      get_outgoing_kwarg_dataflow_outputs_for_node(pattern.raw_graph, node.raw_node);

  return map_values(raw_outputs, [](KwargDataflowOutput<TensorSlotName> const &o) {
    return PatternNodeOutput{o};
  });
}

bool assignment_satisfies(SubParallelComputationGraph const &pcg,
                          PCGPattern const &pattern,
                          PCGPatternMatch const &pattern_match) {
  return unlabelled_pattern_does_match(
      get_unlabelled_pattern(pattern),
      pcg.raw_graph,
      get_unlabelled_pattern_match(pattern_match),
      pcg_pattern_criteria(pattern, pcg));
}

} // namespace FlexFlow
