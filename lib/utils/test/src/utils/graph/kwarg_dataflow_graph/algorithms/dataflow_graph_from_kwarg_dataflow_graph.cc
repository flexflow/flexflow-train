#include "utils/graph/kwarg_dataflow_graph/algorithms/dataflow_graph_from_kwarg_dataflow_graph.h"
#include "utils/containers/get_only.h"
#include "utils/containers/require_only_key.h"
#include "utils/containers/reversed.h"
#include "utils/graph/dataflow_graph/algorithms/dataflow_graph_as_dot.h"
#include "utils/graph/dataflow_graph/algorithms/dataflow_graphs_are_isomorphic.h"
#include "utils/graph/dataflow_graph/dataflow_graph.h"
#include "utils/graph/instances/unordered_set_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_as_dot.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dataflow_graph_from_kwarg_dataflow_graph") {

    KwargDataflowGraphView<std::string> input = [] {
      KwargDataflowGraph<std::string> g =
          KwargDataflowGraph<std::string>::template create<
              UnorderedSetKwargDataflowGraph<std::string>>();

      KwargNodeAddedResult<std::string> n0_added = g.add_node(
          /*inputs=*/std::unordered_map<std::string,
                                        KwargDataflowOutput<std::string>>{},
          /*outputs=*/std::unordered_set<std::string>{
              "a",
          });

      KwargDataflowOutput o0 =
          require_only_key(n0_added.outputs, std::string{"a"});

      KwargNodeAddedResult<std::string> n1_added = g.add_node(
          /*inputs=*/std::unordered_map<std::string,
                                        KwargDataflowOutput<std::string>>{},
          /*outputs=*/std::unordered_set<std::string>{
              "b",
              "c",
          });

      KwargDataflowOutput<std::string> o1 = n1_added.outputs.at("b");
      KwargDataflowOutput<std::string> o2 = n1_added.outputs.at("c");

      KwargNodeAddedResult<std::string> n2_added = g.add_node(
          /*inputs=*/
          std::unordered_map<std::string, KwargDataflowOutput<std::string>>{
              {"z", o1},
              {"y", o2},
              {"x", o0},
          },
          /*outputs=*/std::unordered_set<std::string>{
              "d",
          });

      return g;
    }();

    std::function<std::vector<std::string>(
        std::unordered_set<std::string> const &)>
        slot_ordering = [](std::unordered_set<std::string> const &slots)
        -> std::vector<std::string> { return reversed(sorted(slots)); };

    DataflowGraphView result =
        dataflow_graph_from_kwarg_dataflow_graph(input, slot_ordering);

    DataflowGraphView correct = [] {
      DataflowGraph g = DataflowGraph::create<UnorderedSetDataflowGraph>();

      NodeAddedResult n0_added = g.add_node(
          /*inputs=*/{},
          /*num_outputs=*/1_n);

      DataflowOutput o0 = get_only(n0_added.outputs);

      NodeAddedResult n1_added = g.add_node(
          /*inputs=*/{},
          /*num_outputs=*/2_n);

      DataflowOutput o1 = n1_added.outputs.at(0);
      DataflowOutput o2 = n1_added.outputs.at(1);

      NodeAddedResult n2_added = g.add_node(
          /*inputs=*/{o2, o1, o0},
          /*num_outputs=*/1_n);

      return g;
    }();

    CHECK(dataflow_graphs_are_isomorphic(result, correct));
  }
}
