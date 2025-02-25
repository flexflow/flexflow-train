#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_GRAPH_OPTIMIZE_RESULT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_GRAPH_OPTIMIZE_RESULT_H

#include "compiler/search_result.dtg.h"

namespace FlexFlow {

std::string format_as(SearchResult const &);
std::ostream &operator<<(std::ostream &, SearchResult const &);

} // namespace FlexFlow

#endif