#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/rapidcheck.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree") {
    Node n1 = Node{1};
    Node n2 = Node{2};
    Node n3 = Node{3};

    auto make_binary_series_split = [](BinarySPDecompositionTree const &lhs,
                                       BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinarySeriesSplit{lhs, rhs}};
    };

    auto make_binary_parallel_split = [](BinarySPDecompositionTree const &lhs,
                                         BinarySPDecompositionTree const &rhs) {
      return BinarySPDecompositionTree{BinaryParallelSplit{lhs, rhs}};
    };

    auto make_binary_leaf = [](Node const &n) {
      return BinarySPDecompositionTree{n};
    };

    auto make_pcg_series_split = [](PCGBinarySPDecomposition const &lhs,
                                    PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinarySeriesSplit{lhs, rhs}};
    };

    auto make_pcg_parallel_split = [](PCGBinarySPDecomposition const &lhs,
                                      PCGBinarySPDecomposition const &rhs) {
      return PCGBinarySPDecomposition{PCGBinaryParallelSplit{lhs, rhs}};
    };

    auto make_pcg_leaf = [](Node const &n) {
      return PCGBinarySPDecomposition{parallel_layer_guid_t{n}};
    };

    SUBCASE("single node") {
      BinarySPDecompositionTree input = make_binary_leaf(n1);

      PCGBinarySPDecomposition result =
          pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(input);

      PCGBinarySPDecomposition expected = make_pcg_leaf(n1);

      CHECK(result == expected);
    }

    SUBCASE("series split") {
      BinarySPDecompositionTree input =
          make_binary_series_split(make_binary_leaf(n1), make_binary_leaf(n2));

      PCGBinarySPDecomposition result =
          pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(input);

      PCGBinarySPDecomposition expected =
          make_pcg_series_split(make_pcg_leaf(n1), make_pcg_leaf(n2));

      CHECK(result == expected);
    }

    SUBCASE("parallel split") {
      BinarySPDecompositionTree input = make_binary_parallel_split(
          make_binary_leaf(n1), make_binary_leaf(n2));

      PCGBinarySPDecomposition result =
          pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(input);

      PCGBinarySPDecomposition expected =
          make_pcg_parallel_split(make_pcg_leaf(n1), make_pcg_leaf(n2));

      CHECK(result == expected);
    }

    SUBCASE("bijectiveness") {
      BinarySPDecompositionTree original = make_binary_parallel_split(
          make_binary_series_split(make_binary_leaf(n1), make_binary_leaf(n2)),
          make_binary_leaf(n3));

      PCGBinarySPDecomposition pcg_tree =
          pcg_binary_sp_decomposition_from_binary_sp_decomposition_tree(original);
      BinarySPDecompositionTree converted =
          binary_sp_tree_from_pcg_sp_tree(pcg_tree);

      CHECK(original == converted);
    }
  }
}
