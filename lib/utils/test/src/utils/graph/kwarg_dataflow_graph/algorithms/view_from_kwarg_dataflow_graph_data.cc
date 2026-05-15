#include "utils/graph/kwarg_dataflow_graph/algorithms/view_from_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_edges.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("view_from_kwarg_dataflow_graph_data") {
    auto mk_edge = [](Node src,
                      std::optional<int> src_idx,
                      Node dst,
                      std::optional<int> dst_idx)
        -> KwargDataflowEdge<std::optional<int>> {
      return KwargDataflowEdge<std::optional<int>>{
          /*src=*/KwargDataflowOutput<std::optional<int>>{
              /*node=*/src,
              /*slot_name=*/src_idx,
          },
          /*dst=*/
          KwargDataflowInput<std::optional<int>>{
              /*node=*/dst,
              /*slot_name=*/dst_idx,
          },
      };
    };

    auto mk_output = [](Node src, std::optional<int> src_idx)
        -> KwargDataflowOutput<std::optional<int>> {
      return KwargDataflowOutput<std::optional<int>>{
          /*node=*/src,
          /*slot_name=*/src_idx,
      };
    };

    Node n0 = Node{0};
    Node n1 = Node{1};
    Node n2 = Node{2};

    std::unordered_set<Node> all_nodes = {n0, n1, n2};

    std::unordered_set<KwargDataflowEdge<std::optional<int>>> all_edges = {
        mk_edge(n0, 1, n1, 0),
        mk_edge(n0, 1, n1, std::nullopt),
        mk_edge(n1, 2, n2, 3),
        mk_edge(n0, std::nullopt, n2, 1),
    };

    std::unordered_set<KwargDataflowOutput<std::optional<int>>> all_outputs = {
        mk_output(n0, 1),
        mk_output(n0, std::nullopt),
        mk_output(n0, 4),
        mk_output(n1, 2),
        mk_output(n2, 4),
    };

    KwargDataflowGraphData<std::optional<int>> data =
        KwargDataflowGraphData<std::optional<int>>{
            /*nodes=*/{n0, n1, n2},
            /*edges=*/all_edges,
            /*outputs=*/all_outputs,
        };

    KwargDataflowGraphView<std::optional<int>> g =
        view_from_kwarg_dataflow_graph_data(data);

    SUBCASE("get_nodes") {
      std::unordered_set<Node> result = get_nodes(g);
      std::unordered_set<Node> correct = all_nodes;
      ASSERT(result == correct);
    }

    SUBCASE("get_all_kwarg_dataflow_edges") {
      std::unordered_set<KwargDataflowEdge<std::optional<int>>> result =
          get_all_kwarg_dataflow_edges(g);
      std::unordered_set<KwargDataflowEdge<std::optional<int>>> correct =
          all_edges;
      ASSERT(result == correct);
    }

    SUBCASE("get_all_kwarg_dataflow_outputs") {
      std::unordered_set<KwargDataflowOutput<std::optional<int>>> result =
          get_all_kwarg_dataflow_outputs(g);
      std::unordered_set<KwargDataflowOutput<std::optional<int>>> correct =
          all_outputs;
      ASSERT(result == correct);
    }
  }
}
