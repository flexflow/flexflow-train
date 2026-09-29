#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SEARCH_RESULT_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_SEARCH_RESULT_H

#include "compiler/search_result.dtg.h"
#include "pcg/computation_graph.dtg.h"

namespace FlexFlow {

MappedParallelComputationGraph
    get_mapped_pcg_from_search_result(SearchResult const &);

SearchResult trivial_search_result_for_cg(ComputationGraph const &,
                                          MachineSpaceCoordinate const &);

std::string format_as(SearchResult const &);
std::ostream &operator<<(std::ostream &, SearchResult const &);

} // namespace FlexFlow

#endif
