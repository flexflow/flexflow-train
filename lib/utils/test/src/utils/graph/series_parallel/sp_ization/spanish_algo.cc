#include "utils/graph/serial_parallel/sp_ization/spanish_algo.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/serial_parallel/sp_ization/dependencies_are_maintained.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("spanish_algorithm") {

    // SUBCASE("Single Node") {
    //     DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    //     g.add_node();
    //   SerialParallelDecomposition sp = one_node_at_a_time_spanish_sp_ization(g);
    //   CHECK(dependencies_are_maintained(g, sp));
    // }
    // SUBCASE("Linear Graph") {
    //     DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    //     std::vector<Node> n = add_nodes(g, 3);
    //     add_edges(g,
    //             {
    //                 DirectedEdge{n[0], n[1]},
    //                 DirectedEdge{n[1], n[2]}});
    //   SerialParallelDecomposition sp = one_node_at_a_time_spanish_sp_ization(g);

    //   CHECK(dependencies_are_maintained(g, sp));
        
    // }

    SUBCASE("Rhombus") {
        DiGraph g = DiGraph::create<AdjacencyDiGraph>();
        std::vector<Node> n = add_nodes(g, 4);
        add_edges(g,
                {
                    DirectedEdge{n[0], n[1]},
                    DirectedEdge{n[0], n[2]},
                    DirectedEdge{n[1], n[3]},
                    DirectedEdge{n[2], n[3]}}
                    );
      SerialParallelDecomposition sp = one_node_at_a_time_spanish_sp_ization(g);

      CHECK(dependencies_are_maintained(g, sp));
        

    
    }

//     SUBCASE("Sample Graph #1") {
//       DiGraph g = DiGraph::create<AdjacencyDiGraph>();
//       std::vector<Node> n = add_nodes(g, 6);
//       add_edges(g,
//                 {DirectedEdge{n[1], n[2]}},
                    
//                     DirectedEdge{n[0], n[1]},
//                     DirectedEdge{n[0], n[2]},
//                     DirectedEdge{n[1], n[2]},
//                     DirectedEdge{n[1], n[3]},
//                     DirectedEdge{n[2], n[4]},
//                     DirectedEdge{n[3], n[4]},
//                     DirectedEdge{n[3], n[5]},
//                     DirectedEdge{n[4], n[5]},
//                 });
//       SerialParallelDecomposition sp = one_node_at_a_time_spanish_sp_ization(g);
//       CHECK(dependencies_are_maintained(g, sp));
//     }
  }
}
