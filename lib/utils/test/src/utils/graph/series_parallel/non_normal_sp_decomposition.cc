#include "utils/graph/series_parallel/non_normal_sp_decomposition.h"
#include "doctest/doctest.h"
#include "utils/graph/series_parallel/series_parallel_decomposition.dtg.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("is_empty_non_normal(NonNormalSPDecomposition)") {
    Node n1 = Node{1};
    Node n2 = Node{2};

    SUBCASE("Node Decomposition") {
      NonNormalSPDecomposition sp = NonNormalSPDecomposition{n1};
      CHECK_FALSE(is_empty_non_normal(sp));
    }

    SUBCASE("Empty Series") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalSeriesSplit{
              std::vector<std::variant<NonNormalParallelSplit, Node>>{}}};
      CHECK(is_empty_non_normal(sp));
    }

    SUBCASE("Empty Parallel") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalParallelSplit{{}}};
      CHECK(is_empty_non_normal(sp));
    }

    SUBCASE("Series with Node") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalSeriesSplit{{n1}}};
      CHECK_FALSE(is_empty_non_normal(sp));
    }

    SUBCASE("Parallel with Node") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalParallelSplit{{n1}}};
      CHECK_FALSE(is_empty_non_normal(sp));
    }

    SUBCASE("Nested Series") {
      NonNormalSPDecomposition sp = NonNormalSPDecomposition{
          NonNormalSeriesSplit{{NonNormalParallelSplit{{}}}}};
      CHECK(is_empty_non_normal(sp));
    }

    SUBCASE("Nested Parallel") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalParallelSplit{{NonNormalSeriesSplit{
              std::vector<std::variant<NonNormalParallelSplit, Node>>{}}}}};
      CHECK(is_empty_non_normal(sp));
    }

    SUBCASE("Sparse") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalSeriesSplit{
              {NonNormalParallelSplit{{}},
               NonNormalParallelSplit{{NonNormalSeriesSplit{std::vector<
                   std::variant<NonNormalParallelSplit, Node>>{}}}}}}};
      CHECK(is_empty_non_normal(sp));
    }

    SUBCASE("Sparse with Node") {
      NonNormalSPDecomposition sp =
          NonNormalSPDecomposition{NonNormalSeriesSplit{
              {NonNormalParallelSplit{{}},
               NonNormalParallelSplit{
                   {NonNormalSeriesSplit{std::vector<
                        std::variant<NonNormalParallelSplit, Node>>{}},
                    n2}}}}};
      CHECK_FALSE(is_empty_non_normal(sp));
    }
  }

  TEST_CASE("as_non_normal(SeriesParallelDecomposition)") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};
    Node n4 = Node{4};

    SUBCASE("Node") {
      SeriesParallelDecomposition input = SeriesParallelDecomposition{n1};
      NonNormalSPDecomposition result = as_non_normal(input);
      NonNormalSPDecomposition correct = NonNormalSPDecomposition{n1};
      CHECK(result == correct);
    }

    SUBCASE("SeriesSplit with Nodes") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{SeriesSplit{{n1, n2, n3}}};
      NonNormalSPDecomposition result = as_non_normal(input);
      NonNormalSPDecomposition correct =
          NonNormalSPDecomposition{NonNormalSeriesSplit{{n1, n2, n3}}};
      CHECK(result == correct);
    }

    SUBCASE("ParallelSplit with Nodes") {
      SeriesParallelDecomposition input =
          SeriesParallelDecomposition{ParallelSplit{{n1, n2}}};
      NonNormalSPDecomposition result = as_non_normal(input);
      NonNormalSPDecomposition correct =
          NonNormalSPDecomposition{NonNormalParallelSplit{{n1, n2}}};
      CHECK(result == correct);
    }

    SUBCASE("SeriesSplit containing ParallelSplit") {
      // S(P(n1, n2), n3)
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{{n1, n2}}, n3}}};
      NonNormalSPDecomposition result = as_non_normal(input);
      NonNormalSPDecomposition correct = NonNormalSPDecomposition{
          NonNormalSeriesSplit{{NonNormalParallelSplit{{n1, n2}}, n3}}};
      CHECK(result == correct);
    }

    SUBCASE("ParallelSplit containing SeriesSplit") {
      // P(S(n1, n2), n3)
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          ParallelSplit{{SeriesSplit{{n1, n2}}, n3}}};
      NonNormalSPDecomposition result = as_non_normal(input);
      NonNormalSPDecomposition correct = NonNormalSPDecomposition{
          NonNormalParallelSplit{{NonNormalSeriesSplit{{n1, n2}}, n3}}};
      CHECK(result == correct);
    }

    SUBCASE("deeply nested") {
      // S(P(S(n1, n2), n3), n4)
      SeriesParallelDecomposition input = SeriesParallelDecomposition{
          SeriesSplit{{ParallelSplit{{SeriesSplit{{n1, n2}}, n3}}, n4}}};
      NonNormalSPDecomposition result = as_non_normal(input);
      NonNormalSPDecomposition correct =
          NonNormalSPDecomposition{NonNormalSeriesSplit{
              {NonNormalParallelSplit{{NonNormalSeriesSplit{{n1, n2}}, n3}},
               n4}}};
      CHECK(result == correct);
    }
  }
}
