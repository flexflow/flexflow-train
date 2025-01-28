#ifndef _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_STANDARD_PATTERN_EDGE_H
#define _FLEXFLOW_LIB_SUBSTITUTIONS_INCLUDE_SUBSTITUTIONS_UNLABELLED_STANDARD_PATTERN_EDGE_H

#include "substitutions/unlabelled/pattern_node.dtg.h"
#include "substitutions/unlabelled/standard_pattern_edge.dtg.h"

namespace FlexFlow {

PatternNode get_src_node(StandardPatternEdge const &);
PatternNode get_dst_node(StandardPatternEdge const &);
nonnegative_int get_src_idx(StandardPatternEdge const &);
nonnegative_int get_dst_idx(StandardPatternEdge const &);

} // namespace FlexFlow

#endif
