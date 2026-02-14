#include "utils/graph/series_parallel/sp_ization/node_role.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_edges.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/node/node.dtg.h"
#include "utils/graph/series_parallel/sp_ization/node_role.dtg.h"
#include <doctest/doctest.h>
#include <unordered_set>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("contract_out_nodes_of_given_role") {
    SUBCASE("contract out dummy nodes") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 5);
      std::vector<DirectedEdge> edges = {
          DirectedEdge{n.at(0), n.at(1)},
          DirectedEdge{n.at(1), n.at(2)},
          DirectedEdge{n.at(1), n.at(3)},
          DirectedEdge{n.at(2), n.at(4)},
          DirectedEdge{n.at(3), n.at(4)},
      };
      add_edges(g, edges);

      std::unordered_map<Node, NodeRole> node_roles = {
          {n.at(0), NodeRole::PURE},
          {n.at(1), NodeRole::DUMMY},
          {n.at(2), NodeRole::DUMMY},
          {n.at(3), NodeRole::PURE},
          {n.at(4), NodeRole::PURE},
      };

      DiGraph result =
          contract_out_nodes_of_given_role(g, NodeRole::DUMMY, node_roles);

      CHECK(get_nodes(result) ==
            std::unordered_set<Node>{n.at(0), n.at(3), n.at(4)});
      CHECK(get_edges(result) ==
            std::unordered_set<DirectedEdge>{DirectedEdge{n.at(0), n.at(4)},
                                             DirectedEdge{n.at(0), n.at(3)},
                                             DirectedEdge{n.at(3), n.at(4)}});
    }

    SUBCASE("contract out sync nodes") {
      DiGraph g = DiGraph::create<AdjacencyDiGraph>();
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(1), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)}});

      std::unordered_map<Node, NodeRole> node_roles = {
          {n.at(0), NodeRole::PURE},
          {n.at(1), NodeRole::SYNC},
          {n.at(2), NodeRole::PURE},
          {n.at(3), NodeRole::PURE},
      };

      DiGraph result =
          contract_out_nodes_of_given_role(g, NodeRole::SYNC, node_roles);

      CHECK(get_nodes(result) ==
            std::unordered_set<Node>{n.at(0), n.at(2), n.at(3)});
      CHECK(get_edges(result) ==
            std::unordered_set<DirectedEdge>{DirectedEdge{n.at(0), n.at(2)},
                                             DirectedEdge{n.at(0), n.at(3)}});
    }
  }
}
