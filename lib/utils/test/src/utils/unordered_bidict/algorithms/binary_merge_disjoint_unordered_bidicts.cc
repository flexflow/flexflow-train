#include "utils/unordered_bidict/algorithms/binary_merge_disjoint_unordered_bidicts.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("binary_merge_disjoint_unordered_bidicts") {

    SUBCASE("disjoint keys and values") {
      unordered_bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd2 = {{3, "three"}, {4, "four"}};

      unordered_bidict<int, std::string> result = binary_merge_disjoint_unordered_bidicts(bd1, bd2);
      unordered_bidict<int, std::string> correct = {
          {1, "one"}, {2, "two"}, {3, "three"}, {4, "four"}};

      CHECK(result == correct);
    }

    SUBCASE("overlapping key, different associated value") {
      unordered_bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd2 = {{2, "three"}, {3, "four"}};

      CHECK_THROWS(binary_merge_disjoint_unordered_bidicts(bd1, bd2));
    }

    SUBCASE("overlapping key, same associated value") {
      unordered_bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd2 = {{2, "two"}, {3, "three"}};

      CHECK_THROWS(binary_merge_disjoint_unordered_bidicts(bd1, bd2));
    }

    SUBCASE("overlapping values") {
      unordered_bidict<int, std::string> bd1 = {{1, "one"}, {2, "two"}};
      unordered_bidict<int, std::string> bd2 = {{3, "two"}, {4, "four"}};

      CHECK_THROWS(binary_merge_disjoint_unordered_bidicts(bd1, bd2));
    }
  }
}
