#include "utils/graph/series_parallel/normalize_sp_decomposition.h"
#include "utils/graph/series_parallel/non_normal_parallel_split.dtg.h"
#include "utils/graph/series_parallel/non_normal_series_split.dtg.h"
#include "utils/graph/series_parallel/non_normal_sp_decomposition.dtg.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("normalize_sp_decomposition") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    SUBCASE("Empty") {
      NonNormalSPDecomposition input = NonNormalSPDecomposition{
          NonNormalSeriesSplit{
              {NonNormalParallelSplit{{}}, NonNormalParallelSplit{{}}}}};
      CHECK_THROWS_AS(normalize_sp_decomposition(input), std::runtime_error);
    }

    SUBCASE("Node Decomposition") {
      NonNormalSPDecomposition input = NonNormalSPDecomposition{n1};
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Series with Single Node") {
      NonNormalSPDecomposition input =
          NonNormalSPDecomposition{NonNormalSeriesSplit{{n1}}};
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Parallel with Single Node") {
      NonNormalSPDecomposition input =
          NonNormalSPDecomposition{NonNormalParallelSplit{{n1}}};
      SeriesParallelDecomposition correct = SeriesParallelDecomposition{n1};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Mixed Series") {
      NonNormalSPDecomposition input = NonNormalSPDecomposition{
          NonNormalSeriesSplit{{NonNormalParallelSplit{{n1}}, n2}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{SeriesSplit{{n1, n2}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Mixed Parallel") {
      NonNormalSPDecomposition input = NonNormalSPDecomposition{
          NonNormalParallelSplit{{NonNormalSeriesSplit{{n1}}, n2}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n1, n2}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }

    SUBCASE("Nested") {
      NonNormalSPDecomposition input = NonNormalSPDecomposition{
          NonNormalParallelSplit{{NonNormalSeriesSplit{
                                      {NonNormalParallelSplit{{n1, n2}}}},
                                  n3,
                                  NonNormalSeriesSplit{{}}}}};
      SeriesParallelDecomposition correct =
          SeriesParallelDecomposition{ParallelSplit{{n1, n2, n3}}};
      SeriesParallelDecomposition result = normalize_sp_decomposition(input);
      CHECK(correct == result);
    }
  }
}
