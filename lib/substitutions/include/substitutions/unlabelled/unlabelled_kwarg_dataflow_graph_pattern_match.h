#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_KWARG_DATAFLOW_GRAPH_PATTERN_MATCH_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_UNLABELLED_KWARG_DATAFLOW_GRAPH_PATTERN_MATCH_H

#include "substitutions/pcg_pattern.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/unlabelled/pattern_value.dtg.h"
#include "substitutions/unlabelled/unlabelled_kwarg_dataflow_graph_pattern_match.dtg.h"
#include <optional>
#include <unordered_set>

namespace FlexFlow {

UnlabelledKwargDataflowGraphPatternMatch empty_unlabelled_pattern_match();
std::unordered_set<Node>
    matched_nodes(UnlabelledKwargDataflowGraphPatternMatch const &);
std::optional<UnlabelledKwargDataflowGraphPatternMatch>
    merge_unlabelled_dataflow_graph_pattern_matches(
        UnlabelledKwargDataflowGraphPatternMatch const &subpattern_1,
        UnlabelledKwargDataflowGraphPatternMatch const &subpattern_2,
        bidict<PatternValue, PatternInput> const
            &merged_graph_values_to_inputs_of_1,
        bidict<PatternValue, PatternInput> const
            &merged_graph_values_to_inputs_of_2);

std::unordered_map<OpenKwargDataflowValue<int, TensorSlotName>, PatternValue>
    get_output_assignment(SubParallelComputationGraph const &,
                          PCGPattern const &,
                          UnlabelledKwargDataflowGraphPatternMatch const &);

} // namespace FlexFlow

#endif
