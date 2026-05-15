#include "utils/graph/dataflow_graph/dataflow_edge_query.h"

namespace FlexFlow {

DataflowEdgeQuery dataflow_edge_query_all() {
  return DataflowEdgeQuery{
      query_set<Node>::matchall(),
      query_set<nonnegative_int>::matchall(),
      query_set<Node>::matchall(),
      query_set<nonnegative_int>::matchall(),
  };
}

DataflowEdgeQuery dataflow_edge_query_none() {
  return DataflowEdgeQuery{
      query_set<Node>::match_none(),
      query_set<nonnegative_int>::match_none(),
      query_set<Node>::match_none(),
      query_set<nonnegative_int>::match_none(),
  };
}

bool dataflow_edge_query_includes_dataflow_edge(DataflowEdgeQuery const &q,
                                                DataflowEdge const &e) {
  return includes(q.src_nodes, e.src.node) &&
         includes(q.dst_nodes, e.dst.node) && includes(q.src_idxs, e.src.idx) &&
         includes(q.dst_idxs, e.dst.idx);
}

DataflowEdgeQuery dataflow_edge_query_for_edge(DataflowEdge const &e) {
  return DataflowEdgeQuery{
      query_set<Node>::match_single_value(e.src.node),
      query_set<nonnegative_int>::match_single_value(e.src.idx),
      query_set<Node>::match_single_value(e.dst.node),
      query_set<nonnegative_int>::match_single_value(e.dst.idx),
  };
}

DataflowEdgeQuery
    dataflow_edge_query_all_outgoing_from(DataflowOutput const &src) {
  return DataflowEdgeQuery{
      query_set<Node>::match_single_value(src.node),
      query_set<nonnegative_int>::match_single_value(src.idx),
      query_set<Node>::matchall(),
      query_set<nonnegative_int>::matchall(),
  };
}

DataflowEdgeQuery
    dataflow_edge_query_all_incoming_to(DataflowInput const &dst) {
  return DataflowEdgeQuery{
      query_set<Node>::matchall(),
      query_set<nonnegative_int>::matchall(),
      query_set<Node>::match_single_value(dst.node),
      query_set<nonnegative_int>::match_single_value(dst.idx),
  };
}

} // namespace FlexFlow
