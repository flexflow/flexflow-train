#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/containers/enumerate_vector.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/values.h"
#include "utils/graph/dataflow_graph/dataflow_output_query.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_dataflow_graph/algorithms/from_open_dataflow_graph_data.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_edges.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_incoming_edges.h"
#include "utils/graph/open_dataflow_graph/algorithms/get_subgraph_inputs.h"
#include "utils/graph/open_dataflow_graph/dataflow_graph_input_source.h"
#include "utils/graph/open_dataflow_graph/open_dataflow_value.dtg.h"
#include "utils/hash/pair.h"
#include "utils/overload.h"

namespace FlexFlow {

OpenDataflowSubgraphResult
    get_subgraph(OpenDataflowGraphView const &g,
                 std::unordered_set<Node> const &subgraph_nodes) {
  bidict<OpenDataflowValue, DataflowGraphInput>
      full_graph_values_to_subgraph_inputs =
          get_full_graph_values_to_subgraph_inputs(g, subgraph_nodes);

  return OpenDataflowSubgraphResult{
      OpenDataflowGraphView::create<FromOpenDataflowGraphDataView>(
          get_subgraph_data(
              g, subgraph_nodes, full_graph_values_to_subgraph_inputs)),
      full_graph_values_to_subgraph_inputs,
  };
}

bidict<OpenDataflowValue, DataflowGraphInput>
    get_full_graph_values_to_subgraph_inputs(
        OpenDataflowGraphView const &g,
        std::unordered_set<Node> const &subgraph_nodes) {
  DataflowGraphInputSource input_source;
  return generate_bidict(get_subgraph_inputs(g, subgraph_nodes),
                         [&](OpenDataflowValue const &v) -> DataflowGraphInput {
                           return v.visit<DataflowGraphInput>(overload{
                               [](DataflowGraphInput const &i) { return i; },
                               [&](DataflowOutput const &) {
                                 return input_source.new_dataflow_graph_input();
                               },
                           });
                         });
}

OpenDataflowGraphData
    get_subgraph_data(OpenDataflowGraphView const &g,
                      std::unordered_set<Node> const &subgraph_nodes,
                      bidict<OpenDataflowValue, DataflowGraphInput> const
                          &full_graph_values_to_subgraph_inputs) {
  std::unordered_set<OpenDataflowEdge> subgraph_input_edges =
      transform(get_subgraph_incoming_edges(g, subgraph_nodes),
                [&](OpenDataflowEdge const &edge) {
                  return edge.visit<OpenDataflowEdge>(
                      overload{[&](DataflowInputEdge const &e) {
                                 return OpenDataflowEdge{DataflowInputEdge{
                                     full_graph_values_to_subgraph_inputs.at_l(
                                         OpenDataflowValue{e.src}),
                                     e.dst}};
                               },
                               [&](DataflowEdge const &e) {
                                 return OpenDataflowEdge{DataflowInputEdge{
                                     full_graph_values_to_subgraph_inputs.at_l(
                                         OpenDataflowValue{e.src}),
                                     e.dst}};
                               }});
                });

  OpenDataflowEdgeQuery subgraph_interior_edges_query = OpenDataflowEdgeQuery{
      DataflowInputEdgeQuery{
          query_set<DataflowGraphInput>::match_none(),
          query_set<Node>::match_none(),
          query_set<nonnegative_int>::match_none(),
      },
      DataflowEdgeQuery{
          query_set<Node>{subgraph_nodes},
          query_set<nonnegative_int>::matchall(),
          query_set<Node>{subgraph_nodes},
          query_set<nonnegative_int>::matchall(),
      },
  };
  std::unordered_set<OpenDataflowEdge> subgraph_interior_edges =
      g.query_edges(subgraph_interior_edges_query);

  std::unordered_set<DataflowGraphInput> subgraph_inputs =
      unordered_set_of(values(full_graph_values_to_subgraph_inputs));
  std::unordered_set<DataflowOutput> subgraph_outputs =
      filter(g.query_outputs(dataflow_output_query_all()),
             [&](DataflowOutput const &o) {
               return contains(subgraph_nodes, o.node);
             });
  return OpenDataflowGraphData{
      subgraph_nodes,
      set_union(subgraph_input_edges, subgraph_interior_edges),
      subgraph_inputs,
      subgraph_outputs,
  };
}

} // namespace FlexFlow
