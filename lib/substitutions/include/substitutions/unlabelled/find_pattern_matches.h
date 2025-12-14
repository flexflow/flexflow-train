#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_FIND_PATTERN_MATCHES_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_FIND_PATTERN_MATCHES_H

#include "substitutions/unlabelled/match_additional_criterion.dtg.h"
#include "substitutions/unlabelled/unlabelled_kwarg_dataflow_graph_pattern_match.dtg.h"
#include "substitutions/unlabelled/unlabelled_graph_pattern.dtg.h"

namespace FlexFlow {

std::vector<UnlabelledKwargDataflowGraphPatternMatch>
  find_unlabelled_pattern_matches(
    UnlabelledGraphPattern const &pattern,
    OpenKwargDataflowGraphView<int, TensorSlotName> const &graph,
    MatchAdditionalCriterion const &additional_criterion);

} // namespace FlexFlow

#endif
