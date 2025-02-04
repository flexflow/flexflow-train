#ifndef _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H
#define _FLEXFLOW_SUBSTITUTIONS_SUBSTITUTION_H

#include "substitutions/substitution.dtg.h"

namespace FlexFlow {

bool is_isomorphic_to(Substitution const &, Substitution const &);

std::string as_dot(Substitution const &);

/**
 * @brief Checks that all internal invariants of the given substitution hold
 *
 * @details In order for the result of substitution application to be a valid
 * PCG, a Substitution must maintain invariants on the inputs and outputs of
 * both its left-hand side (Substitution::pcg_pattern) and its right-hand side
 * (Substitution::output_graph_expr). More concretely, every Substitution has
 * fields Substitution::input_edge_match_to_output and
 * Substitution::output_edge_match_to_output which must provide a bijection all
 * of the inputs (outputs respectively) of Substitution::pcg_pattern and
 * Substitution::output_graph_expr. If any of these invariants are violated,
 * this function returns false instead of true.
 */
bool is_valid_substitution(Substitution const &);

} // namespace FlexFlow

#endif
