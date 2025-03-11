#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_GRAPH_OPTIMIZE_RESULT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_GRAPH_OPTIMIZE_RESULT_H

#include "compiler/graph_optimize_result.dtg.h"

namespace FlexFlow {

std::string format_as(GraphOptimizeResult const &);
std::ostream &operator<<(std::ostream &, GraphOptimizeResult const &);

} // namespace FlexFlow

#endif
