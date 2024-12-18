#include "utils/graph/serial_parallel/get_ancestors.h"
#include "utils/graph/serial_parallel/serial_parallel_decomposition.dtg.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_ancestors") {
    std::vector<Node> n = {
        Node(0), Node(1), Node(2), Node(3), Node(4), Node(5), Node(6), Node(7)};

    SUBCASE("Single Node") {
      SerialParallelDecomposition sp = SerialParallelDecomposition{n.at(0)};
      std::unordered_set<Node> correct = {};
      std::unordered_set<Node> result = get_ancestors(sp, n.at(0));
      CHECK(correct == result);
    }

    SUBCASE("Simple Serial") {
      SerialParallelDecomposition sp =
          SerialParallelDecomposition{SerialSplit{n.at(0), n.at(1), n.at(2)}};
      std::unordered_set<Node> correct = {n.at(0), n.at(1)};
      std::unordered_set<Node> result = get_ancestors(sp, n.at(2));
      CHECK(correct == result);
    }

    SUBCASE("Simple Parallel") {
      SerialParallelDecomposition sp =
          SerialParallelDecomposition{ParallelSplit{n.at(0), n.at(1), n.at(2)}};
      std::unordered_set<Node> correct = {};
      std::unordered_set<Node> result = get_ancestors(sp, n.at(1));
      CHECK(correct == result);
    }

    SUBCASE("Tree") {
      SerialParallelDecomposition sp = SerialParallelDecomposition{SerialSplit{
          n.at(0), ParallelSplit{SerialSplit{n.at(1), n.at(2)}, n.at(3)}}};
      std::unordered_set<Node> correct = {n.at(0), n.at(1)};
      std::unordered_set<Node> result = get_ancestors(sp, n.at(2));
      CHECK(correct == result);
    }

    SUBCASE("Rhombus") {
      SerialParallelDecomposition sp = SerialParallelDecomposition{
          SerialSplit{n.at(0), ParallelSplit{n.at(1), n.at(2)}, n.at(3)}};
      std::unordered_set<Node> correct = {n.at(0), n.at(1), n.at(2)};
      std::unordered_set<Node> result = get_ancestors(sp, n.at(3));
      CHECK(correct == result);
    }

    SUBCASE("Complex Structure") {
      SerialParallelDecomposition sp = SerialParallelDecomposition{SerialSplit{
          n.at(0),
          ParallelSplit{
              SerialSplit{n.at(1), ParallelSplit{n.at(2), n.at(3)}, n.at(4)},
              SerialSplit{n.at(5), n.at(6)}},
          n.at(7)}};
      std::unordered_set<Node> correct = {n.at(0), n.at(1), n.at(2), n.at(3)};
      std::unordered_set<Node> result = get_ancestors(sp, n.at(4));
      CHECK(correct == result);

      correct = {n.at(0), n.at(1)};
      result = get_ancestors(sp, n.at(3));
      CHECK(correct == result);

      correct = {n.at(0), n.at(5)};
      result = get_ancestors(sp, n.at(6));
      CHECK(correct == result);

      correct = {n.at(0), n.at(1), n.at(2), n.at(3), n.at(4), n.at(5), n.at(6)};
      result = get_ancestors(sp, n.at(7));
      CHECK(correct == result);
    }
  }
}
