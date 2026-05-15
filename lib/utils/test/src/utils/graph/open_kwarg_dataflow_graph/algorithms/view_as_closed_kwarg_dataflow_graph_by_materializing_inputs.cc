#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_as_closed_kwarg_dataflow_graph_by_materializing_inputs.h"
#include "utils/containers/require_only_key.h"
#include "utils/graph/instances/unordered_set_kwarg_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_open_kwarg_dataflow_graph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_inputs.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_all_kwarg_dataflow_outputs.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graphs_are_isomorphic.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("view_as_closed_kwarg_dataflow_graph_by_materializing_inputs") {
    OpenKwargDataflowGraphView<std::string, int> open_g = [] {
      OpenKwargDataflowGraph<std::string, int> g =
          OpenKwargDataflowGraph<std::string, int>::template create<
              UnorderedSetOpenKwargDataflowGraph<std::string, int>>();

      KwargDataflowGraphInput<std::string> input1 = g.add_input("input1");
      KwargDataflowGraphInput<std::string> input2 = g.add_input("input2");

      KwargNodeAddedResult<int> n1_added = g.add_node(
          /*inputs=*/
          std::unordered_map<int, OpenKwargDataflowValue<std::string, int>>{
              {
                  1,
                  OpenKwargDataflowValue<std::string, int>{input1},
              },
              {
                  3,
                  OpenKwargDataflowValue<std::string, int>{input2},
              },
              {
                  8,
                  OpenKwargDataflowValue<std::string, int>{input1},
              },
          },
          /*outputs=*/std::unordered_set<int>{
              5,
          });

      KwargDataflowOutput<int> n1_output =
          require_only_key(n1_added.outputs, 5);

      KwargNodeAddedResult<int> n2_added = g.add_node(
          /*inputs=*/
          std::unordered_map<int, OpenKwargDataflowValue<std::string, int>>{
              {
                  4,
                  OpenKwargDataflowValue<std::string, int>{input2},
              },
              {
                  1,
                  OpenKwargDataflowValue<std::string, int>{n1_output},
              },
          },
          /*outputs=*/std::unordered_set<int>{
              5,
          });

      KwargDataflowOutput<int> n2_output =
          require_only_key(n2_added.outputs, 5);

      return g;
    }();

    std::pair<KwargDataflowGraphView<std::optional<int>>,
              bidict<KwargDataflowGraphInput<std::string>, Node>>
        result =
            view_as_closed_kwarg_dataflow_graph_by_materializing_inputs(open_g);

    KwargDataflowGraphView<std::optional<int>> result_g = result.first;

    KwargDataflowGraphView<std::optional<int>> correct = [] {
      KwargDataflowGraph<std::optional<int>> g =
          KwargDataflowGraph<std::optional<int>>::template create<
              UnorderedSetKwargDataflowGraph<std::optional<int>>>();

      KwargNodeAddedResult<std::optional<int>> input1_added = g.add_node(
          /*inputs=*/{},
          /*outputs=*/std::unordered_set<std::optional<int>>{
              std::nullopt,
          });

      KwargDataflowOutput<std::optional<int>> input1 = require_only_key(
          input1_added.outputs, std::optional<int>{std::nullopt});

      KwargNodeAddedResult<std::optional<int>> input2_added = g.add_node(
          /*inputs=*/{},
          /*outputs=*/std::unordered_set<std::optional<int>>{
              std::nullopt,
          });

      KwargDataflowOutput<std::optional<int>> input2 = require_only_key(
          input2_added.outputs, std::optional<int>{std::nullopt});

      KwargNodeAddedResult<std::optional<int>> n1_added = g.add_node(
          /*inputs=*/
          std::unordered_map<std::optional<int>,
                             KwargDataflowOutput<std::optional<int>>>{
              {
                  1,
                  input1,
              },
              {
                  3,
                  input2,
              },
              {
                  8,
                  input1,
              },
          },
          /*outputs=*/std::unordered_set<std::optional<int>>{
              5,
          });

      KwargDataflowOutput<std::optional<int>> n1_output =
          require_only_key(n1_added.outputs, std::optional<int>{5});

      KwargNodeAddedResult<std::optional<int>> n2_added = g.add_node(
          /*inputs=*/
          std::unordered_map<std::optional<int>,
                             KwargDataflowOutput<std::optional<int>>>{
              {
                  4,
                  input2,
              },
              {
                  1,
                  n1_output,
              },
          },
          /*outputs=*/std::unordered_set<std::optional<int>>{
              5,
          });

      return g;
    }();

    ASSERT(get_nodes(result_g).size() == 4);
    ASSERT(get_all_kwarg_dataflow_edges(result_g).size() ==
           get_all_open_kwarg_dataflow_edges(open_g).size());
    ASSERT(get_all_kwarg_dataflow_inputs(result_g).size() == 5);
    ASSERT(get_all_kwarg_dataflow_outputs(result_g).size() == 4);
    CHECK(kwarg_dataflow_graphs_are_isomorphic(result_g, correct));
  }
}
