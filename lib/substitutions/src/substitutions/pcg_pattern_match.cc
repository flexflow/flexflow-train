#include "substitutions/pcg_pattern_match.h"
#include "substitutions/pcg_pattern.h"
#include "substitutions/sub_parallel_computation_graph.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.h"
#include "utils/bidict/algorithms/bidict_exhaustive_relational_join.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/bidict/algorithms/bidict_from_map.h"
#include "utils/bidict/algorithms/bidict_transform_values.h"
#include "utils/bidict/algorithms/binary_merge_disjoint_bidicts.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/map_values.h"
#include "utils/containers/values.h"
#include "utils/containers/zip.h"

namespace FlexFlow {

bidict<PatternNodeOutput, parallel_tensor_guid_t>
    get_output_mapping_for_pcg_pattern_match(
        PCGPatternMatch const &match,
        PCGPattern const &pattern,
        SubParallelComputationGraph const &spcg) {
  bidict<PatternNodeOutput, parallel_tensor_guid_t> result;

  for (auto const &[pattern_node, matched_layer] : match.node_assignment) {
    bidict<TensorSlotName, parallel_tensor_guid_t>
        matched_layer_output_tensors =
            bidict_from_map(get_outgoing_tensors(spcg, matched_layer));
    bidict<TensorSlotName, PatternNodeOutput> pattern_node_outputs =
        bidict_from_map(get_pattern_node_outputs(pattern, pattern_node));

    assert(matched_layer_output_tensors.size() == pattern_node_outputs.size());

    bidict<PatternNodeOutput, parallel_tensor_guid_t> mapping =
        bidict_exhaustive_relational_join(pattern_node_outputs.reversed(),
                                          matched_layer_output_tensors);

    result = binary_merge_disjoint_bidicts(result, mapping);
  }

  return result;
}

UnlabelledKwargDataflowGraphPatternMatch
    get_unlabelled_pattern_match(PCGPatternMatch const &match) {
  return UnlabelledKwargDataflowGraphPatternMatch{
      bidict_transform_values(
          match.node_assignment,
          [](parallel_layer_guid_t const &l) { return l.raw_graph_node; }),
      map_values(match.input_assignment,
                 [](open_parallel_tensor_guid_t const &i) {
                   return i.raw_open_dataflow_value;
                 }),
  };
}

void assert_pcg_pattern_match_is_valid_for_pattern_and_subpcg(
    PCGPatternMatch const &match,
    PCGPattern const &pattern,
    SubParallelComputationGraph const &spcg) {
  std::set<parallel_layer_guid_t> spcg_nodes = spcg_get_parallel_layers(spcg);
  std::set<parallel_layer_guid_t> match_nodes =
      match.node_assignment.right_values();
  ASSERT(is_subseteq_of(match_nodes, spcg_nodes));

  std::set<open_parallel_tensor_guid_t> spcg_values =
      get_parallel_tensors(spcg);
  std::set<open_parallel_tensor_guid_t> match_values =
      set_of(values(match.input_assignment));
  ASSERT(is_subseteq_of(match_values, spcg_values));

  std::set<PatternNode> pattern_nodes = get_nodes(pattern);
  std::set<PatternNode> match_pattern_nodes =
      match.node_assignment.left_values();
  ASSERT(match_pattern_nodes == pattern_nodes);

  std::set<PatternInput> pattern_inputs = get_inputs(pattern);
  std::set<PatternInput> match_pattern_inputs = keys(match.input_assignment);
  ASSERT(pattern_inputs == match_pattern_inputs);
}

} // namespace FlexFlow
