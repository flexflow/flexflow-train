#include "utils/graph/series_parallel/sp_ization/dependencies_are_maintained.h"
#include "utils/containers/get_only.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("dependencies_are_maintained") {
    DiGraph g = DiGraph::create<AdjacencyDiGraph>();
    SUBCASE("Single Node") {
      std::vector<Node> n = add_nodes(g, 1);
      SeriesParallelDecomposition sp =
          SeriesParallelDecomposition{SeriesSplit{{n[0]}}};
      CHECK(dependencies_are_maintained(g, sp));
    }

    SUBCASE("SeriesSplit") {
      SUBCASE("Valid SP-ization") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(g, {DirectedEdge{n[0], n[1]}, DirectedEdge{n[1], n[2]}});
        SeriesParallelDecomposition sp =
            SeriesParallelDecomposition{SeriesSplit{{n[0], n[1], n[2]}}};
        CHECK(dependencies_are_maintained(g, sp));
      }

      SUBCASE("Incorrect SP-ization") {
        std::vector<Node> n = add_nodes(g, 3);
        add_edges(g, {DirectedEdge{n[0], n[1]}, DirectedEdge{n[1], n[2]}});

        SeriesParallelDecomposition sp =
            SeriesParallelDecomposition{SeriesSplit{{n[1], n[0], n[2]}}};
        CHECK_FALSE(dependencies_are_maintained(g, sp));
      }
    }

    SUBCASE("ParallelSplit") {
      SUBCASE("Valid SP-ization") {
        std::vector<Node> n = add_nodes(g, 3);
        SeriesParallelDecomposition sp =
            SeriesParallelDecomposition{ParallelSplit{{n[0], n[1], n[2]}}};
        CHECK(dependencies_are_maintained(g, sp));
      }

      SUBCASE("Incorrect SP-ization") {
        std::vector<Node> n = add_nodes(g, 3);

        SeriesParallelDecomposition sp =
            SeriesParallelDecomposition{ParallelSplit{{n[0], n[2]}}};
        CHECK_FALSE(dependencies_are_maintained(g, sp));
      }
    }

    SUBCASE("Rhombus") {
      std::vector<Node> n = add_nodes(g, 4);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(2), n.at(3)}});
      SUBCASE("Valid SP-izations") {
        SeriesParallelDecomposition sp_correct = SeriesParallelDecomposition{
            SeriesSplit{{n.at(0), ParallelSplit{{n.at(1), n.at(2)}}, n.at(3)}}};
        CHECK(dependencies_are_maintained(g, sp_correct));

        sp_correct = SeriesParallelDecomposition{
            SeriesSplit{{n.at(0), n.at(1), n.at(2), n.at(3)}}};
        CHECK(dependencies_are_maintained(g, sp_correct));
      }
      SUBCASE("Invalid SP-ization") {
        SeriesParallelDecomposition sp_incorrect = SeriesParallelDecomposition{
            ParallelSplit{{n.at(0), SeriesSplit{{n.at(1), n.at(3)}}, n.at(2)}}};
        CHECK_FALSE(dependencies_are_maintained(g, sp_incorrect));
      }
    }

    SUBCASE("Diamond") {
      std::vector<Node> n = add_nodes(g, 6);
      add_edges(g,
                {DirectedEdge{n.at(0), n.at(1)},
                 DirectedEdge{n.at(0), n.at(2)},
                 DirectedEdge{n.at(1), n.at(3)},
                 DirectedEdge{n.at(1), n.at(4)},
                 DirectedEdge{n.at(2), n.at(4)},
                 DirectedEdge{n.at(3), n.at(5)},
                 DirectedEdge{n.at(4), n.at(5)}});

      SUBCASE("Valid SP-izations") {

        SeriesParallelDecomposition sp_correct = SeriesParallelDecomposition{
            SeriesSplit{{n.at(0),
                         ParallelSplit{{n.at(1), n.at(2)}},
                         ParallelSplit{{n.at(3), n.at(4)}},
                         n.at(5)}}};
        CHECK(dependencies_are_maintained(g, sp_correct));

        sp_correct = SeriesParallelDecomposition{
            SeriesSplit{{n.at(0),
                         n.at(1),
                         n.at(2),
                         ParallelSplit{{n.at(3), n.at(4)}},
                         n.at(5)}}};
        CHECK(dependencies_are_maintained(g, sp_correct));

        sp_correct = SeriesParallelDecomposition{
            SeriesSplit{{n.at(0),
                         ParallelSplit{{n.at(1), n.at(2)}},
                         n.at(3),
                         n.at(4),
                         n.at(5)}}};
        CHECK(dependencies_are_maintained(g, sp_correct));
      }

      SUBCASE("Invalid SP-izations") {
        SeriesParallelDecomposition sp_correct = SeriesParallelDecomposition{
            SeriesSplit{{n.at(0),
                         ParallelSplit{{n.at(1), n.at(2), n.at(4)}},
                         n.at(3),
                         n.at(5)}}};
        CHECK_FALSE(dependencies_are_maintained(g, sp_correct));
      }
    }
  }
}
