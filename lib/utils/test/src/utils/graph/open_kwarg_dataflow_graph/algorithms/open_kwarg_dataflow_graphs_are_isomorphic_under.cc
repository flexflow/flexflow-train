#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graphs_are_isomorphic_under.h"
#include "utils/bidict/algorithms/bidict_from_keys_and_values.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/containers/vector_of.h"
#include "utils/graph/instances/unordered_set_open_kwarg_dataflow_graph.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("open_kwarg_dataflow_graphs_are_isomorphic_under") {
    auto mk_graph = [] {
      OpenKwargDataflowGraph<std::string, int> g =
          OpenKwargDataflowGraph<std::string, int>::template create<
              UnorderedSetOpenKwargDataflowGraph<std::string, int>>();

      KwargNodeAddedResult<int> n1_added = g.add_node(
          /*inputs=*/{},
          /*outputs=*/{});

      KwargNodeAddedResult<int> n2_added = g.add_node(
          /*inputs=*/{},
          /*outputs=*/{});

      KwargNodeAddedResult<int> n3_added = g.add_node(
          /*inputs=*/{},
          /*outputs=*/{});

      KwargNodeAddedResult<int> n4_added = g.add_node(
          /*inputs=*/{},
          /*outputs=*/{});

      return g;
    };

    OpenKwargDataflowGraphView<std::string, int> lhs = mk_graph();
    OpenKwargDataflowGraphView<std::string, int> rhs = mk_graph();

    std::unordered_set<Node> lhs_nodes = get_nodes(lhs);
    std::unordered_set<Node> rhs_nodes = get_nodes(rhs);

    std::vector<Node> ordered_lhs_nodes = vector_of(lhs_nodes);

    for (std::vector<Node> ordered_rhs_nodes :
         get_all_permutations(rhs_nodes)) {
      OpenKwargDataflowGraphIsomorphism<std::string> iso =
          OpenKwargDataflowGraphIsomorphism<std::string>{
              /*node_mapping=*/bidict_from_keys_and_values(ordered_lhs_nodes,
                                                           ordered_rhs_nodes),
              /*input_mapping=*/{},
          };

      CHECK(open_kwarg_dataflow_graphs_are_isomorphic_under(lhs, rhs, iso));
    };
  }
}
