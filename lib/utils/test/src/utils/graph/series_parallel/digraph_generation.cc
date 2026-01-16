#include "utils/graph/series_parallel/digraph_generation.h"
#include "utils/graph/digraph/algorithms.h"
#include "utils/graph/instances/adjacency_digraph.h"
#include "utils/graph/node/algorithms.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("digraph_from_sp_decomposition") {
    SUBCASE("Empty") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition(ParallelSplit{{}});
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 0);
      CHECK(num_edges(result) == 0);
    }

    SUBCASE("Complex Empty") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition(
          ParallelSplit{{SeriesSplit{{}}, SeriesSplit{{ParallelSplit{{}}}}}});
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 0);
      CHECK(num_edges(result) == 0);
    }

    SUBCASE("Single Node") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition(Node(1));
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 1);
      CHECK(num_edges(result) == 0);
    }

    SUBCASE("Simple SeriesSplit") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{SeriesSplit{{Node(1), Node(2), Node(3)}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 3);
      CHECK(num_edges(result) == 2);
      CHECK(get_initial_nodes(result).size() == 1);
      CHECK(get_terminal_nodes(result).size() == 1);
    }

    SUBCASE("Simple ParallelSplit") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{Node(1), Node(2), Node(3)}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 3);
      CHECK(num_edges(result) == 0);
      CHECK(get_initial_nodes(result).size() == 3);
      CHECK(get_terminal_nodes(result).size() == 3);
    }

    SUBCASE("Mixed Serial-Parallel") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{{Node(1), Node(2)}},
                       ParallelSplit{{Node(3), Node(4)}}}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 4);
      CHECK(get_initial_nodes(result).size() == 2);
      CHECK(get_terminal_nodes(result).size() == 2);
    }

    SUBCASE("Mixed Parallel-Serial") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{SeriesSplit{{Node(1), Node(2)}},
                         SeriesSplit{{Node(3), Node(4)}}}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 2);
      CHECK(get_initial_nodes(result).size() == 2);
      CHECK(get_terminal_nodes(result).size() == 2);
    }

    SUBCASE("Rhombus") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{Node(1), ParallelSplit{{Node(2), Node(3)}}, Node(4)}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 4);
      CHECK(get_initial_nodes(result).size() == 1);
      CHECK(get_terminal_nodes(result).size() == 1);
    }

    SUBCASE("Duplicate Nodes") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{Node(1), ParallelSplit{{Node(1), Node(2)}}, Node(1)}}};
      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 4);
      CHECK(num_edges(result) == 4);
      CHECK(get_initial_nodes(result).size() == 1);
      CHECK(get_terminal_nodes(result).size() == 1);
    }

    SUBCASE("Complex Graph") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{SeriesSplit{
              {ParallelSplit{{SeriesSplit{{ParallelSplit{{Node(1), Node(2)}},
                                           ParallelSplit{{Node(3), Node(4)}},
                                           Node(5)}},
                              SeriesSplit{{Node(6), Node(7)}}}},
               Node(8)}}};

      DiGraph result = digraph_from_sp_decomposition(input);
      CHECK(num_nodes(result) == 8);
      CHECK(num_edges(result) == 9);
      CHECK(get_initial_nodes(result).size() == 3);
      CHECK(get_terminal_nodes(result).size() == 1);
    }
  }
}
