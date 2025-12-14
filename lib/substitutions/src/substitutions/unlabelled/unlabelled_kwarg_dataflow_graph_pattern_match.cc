#include "substitutions/unlabelled/unlabelled_kwarg_dataflow_graph_pattern_match.h"
#include "utils/bidict/try_merge_nondisjoint_bidicts.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/try_merge_nondisjoint_unordered_maps.h"

namespace FlexFlow {

UnlabelledKwargDataflowGraphPatternMatch empty_unlabelled_pattern_match() {
  return UnlabelledKwargDataflowGraphPatternMatch{
      bidict<PatternNode, Node>{},
      bidict<PatternInput, OpenKwargDataflowValue<int, TensorSlotName>>{},
  };
}

std::optional<UnlabelledKwargDataflowGraphPatternMatch>
    merge_unlabelled_dataflow_graph_pattern_matches(
        UnlabelledKwargDataflowGraphPatternMatch const &subpattern_1,
        UnlabelledKwargDataflowGraphPatternMatch const &subpattern_2,
        bidict<PatternValue, PatternInput> const
            &merged_graph_values_to_inputs_of_1,
        bidict<PatternValue, PatternInput> const
            &merged_graph_values_to_inputs_of_2) {
  bidict<PatternNode, Node> merged_node_assignment = ({
    std::optional<bidict<PatternNode, Node>> result =
        try_merge_nondisjoint_bidicts(subpattern_1.node_assignment,
                                      subpattern_2.node_assignment);
    if (!result.has_value()) {
      return std::nullopt;
    }
    result.value();
  });

  std::unordered_map<PatternInput, OpenKwargDataflowValue<int, TensorSlotName>> merged_input_assignment =
      ({
        std::unordered_map<PatternValue, OpenKwargDataflowValue<int, TensorSlotName>>
            lifted_input_assignment_1 = map_keys(
                subpattern_1.input_assignment, [&](PatternInput const &pi1) {
                  return merged_graph_values_to_inputs_of_1.at_r(pi1);
                });
        std::unordered_map<PatternValue, OpenKwargDataflowValue<int, TensorSlotName>>
            lifted_input_assignment_2 = map_keys(
                subpattern_2.input_assignment, [&](PatternInput const &pi2) {
                  return merged_graph_values_to_inputs_of_2.at_r(pi2);
                });
        std::optional<std::unordered_map<PatternValue, OpenKwargDataflowValue<int, TensorSlotName>>>
            merged = try_merge_nondisjoint_unordered_maps(
                lifted_input_assignment_1, lifted_input_assignment_2);
        if (!merged.has_value()) {
          return std::nullopt;
        }
        filtermap_keys(
            merged.value(),
            [](PatternValue const &v) -> std::optional<PatternInput> {
              if (v.has<PatternInput>()) {
                return v.get<PatternInput>();
              } else {
                return std::nullopt;
              }
            });
      });

  return UnlabelledKwargDataflowGraphPatternMatch{
      merged_node_assignment,
      merged_input_assignment,
  };
}

} // namespace FlexFlow
