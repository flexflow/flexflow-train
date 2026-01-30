#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SEARCH_RESULT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SEARCH_RESULT_H

#include "compiler/search_result.dtg.h"

namespace FlexFlow {

MappedParallelComputationGraph
    get_mapped_pcg_from_search_result(SearchResult const &);

std::string format_as(SearchResult const &);
std::ostream &operator<<(std::ostream &, SearchResult const &);

} // namespace FlexFlow

#endif
