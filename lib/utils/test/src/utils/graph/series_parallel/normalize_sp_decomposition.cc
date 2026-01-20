#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("normalize_sp_decomposition") {
    Node n1 = Node(1);
    Node n2 = Node(2);
    Node n3 = Node(3);

    SUBCASE("Empty") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{{}}, ParallelSplit{{}}}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Node Decomposition") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Serial with Single Node") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{SeriesSplit{{n1}}};
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Parallel with Single Node") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{ParallelSplit{{n1}}};
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Mixed Serial") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{SeriesSplit{{ParallelSplit{{n1}}, n2}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n1, n2}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Mixed Parallel") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{ParallelSplit{{SeriesSplit{{n1}}, n2}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n1, n2}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Nested") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{ParallelSplit{
              {SeriesSplit{{ParallelSplit{{n1, n2}}}}, n3, SeriesSplit{{}}}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n1, n2, n3}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }
  }
}
