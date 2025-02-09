#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_OPEN_DATAFLOW_GRAPH_ISOMORPHISM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_DATAFLOW_GRAPH_ALGORITHMS_OPEN_DATAFLOW_GRAPH_ISOMORPHISM_H

#include "utils/graph/open_dataflow_graph/algorithms/open_dataflow_graph_isomorphism.dtg.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"

namespace FlexFlow {

OpenDataflowValue isomorphism_map_r_open_dataflow_value_from_l(
    OpenDataflowGraphIsomorphism const &iso, OpenDataflowValue const &l_value);
OpenDataflowValue isomorphism_map_l_open_dataflow_value_from_r(
    OpenDataflowGraphIsomorphism const &iso, OpenDataflowValue const &r_value);

DataflowOutput isomorphism_map_r_dataflow_output_from_l(
    OpenDataflowGraphIsomorphism const &iso, DataflowOutput const &l_output);
DataflowOutput isomorphism_map_l_dataflow_output_from_r(
    OpenDataflowGraphIsomorphism const &iso, DataflowOutput const &r_output);

} // namespace FlexFlow

#endif
