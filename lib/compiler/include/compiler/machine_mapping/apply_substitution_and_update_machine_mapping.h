#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_APPLY_SUBSTITUTION_APPLY_SUBSTITUTION_AND_UPDATE_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_APPLY_SUBSTITUTION_APPLY_SUBSTITUTION_AND_UPDATE_MACHINE_MAPPING_H

#include "compiler/search_result.dtg.h"
#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/substitution.dtg.h"

namespace FlexFlow {
/**
 * @brief Applies \p substitution to \p mapped_pcg at the location specified by
 * \p match, returning the resulting SearchResult (mapped pcg)
 *
 * @param mapped_pcg
 * @param substitution
 * @param match The location at which to apply substitution. This location in
 * sub_pcg should match substitution's PCGPattern. Likely created by running
 * FlexFlow::find_pattern_matches(PCGPattern const &,
 * SubParallelComputationGraph const &).
 * @return SearchResult A mapped pcg similar to mapped_pcg, but with
 * the subgraph of the pcg specified by match replaced with the result of the
 * output expression of substitution and the machine mapping updated to account
 * for the new output
 */
SearchResult apply_substitution_and_update_machine_mapping(
    SearchResult const &mapped_pcg,
    Substitution const &sub,
    PCGPatternMatch const &match);

} // namespace FlexFlow

#endif
