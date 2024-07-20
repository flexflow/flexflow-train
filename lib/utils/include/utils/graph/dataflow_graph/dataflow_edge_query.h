#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_DATAFLOW_GRAPH_DATAFLOW_EDGE_QUERY_H

#include "utils/graph/dataflow_graph/dataflow_edge.dtg.h"
#include "utils/graph/dataflow_graph/dataflow_edge_query.dtg.h"

namespace FlexFlow {

DataflowEdgeQuery dataflow_edge_query_all();
DataflowEdgeQuery dataflow_edge_query_none();
bool dataflow_edge_query_includes_dataflow_edge(DataflowEdgeQuery const &,
                                                DataflowEdge const &);

} // namespace FlexFlow

#endif