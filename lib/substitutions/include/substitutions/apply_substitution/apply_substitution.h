#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_APPLY_SUBSTITUTION_APPLY_SUBSTITUTION_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_APPLY_SUBSTITUTION_APPLY_SUBSTITUTION_H

#include "substitutions/pcg_pattern_match.dtg.h"
#include "substitutions/sub_parallel_computation_graph.dtg.h"
#include "substitutions/substitution.dtg.h"

namespace FlexFlow {

/**
 * @brief Applies \p substitution to \p sub_pcg at the location specified by \p
 * match, returning the resulting SubParallelComputationGraph
 *
 * @param sub_pcg
 * @param substitution
 * @param match The location at which to apply substitution. This location in
 * sub_pcg should match substitution's PCGPattern. Likely created by running
 * FlexFlow::find_pattern_matches(PCGPattern const &,
 * SubParallelComputationGraph const &).
 * @return SubParallelComputationGraph A sub-PCG similar to sub_pcg, but with
 * the subgraph specified by match replaced with the result of the output
 * expression of substitution
 */
SubParallelComputationGraph
    apply_substitution(SubParallelComputationGraph const &sub_pcg,
                       Substitution const &substitution,
                       PCGPatternMatch const &match);

} // namespace FlexFlow

#endif
