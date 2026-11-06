#include "utils/unordered_bidict/algorithms/unordered_bidict_from_enumerating.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/unordered_bidict/algorithms/unordered_bidict_left_entries.h"
#include "utils/unordered_bidict/algorithms/unordered_bidict_right_entries.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_from_enumerating(std::vector<T>)") {
    SUBCASE("input has no duplicates") {
      std::vector<int> input = {3, 6, 5, 2};

      unordered_bidict<nonnegative_int, int> result =
          unordered_bidict_from_enumerating(input);

      unordered_bidict<nonnegative_int, int> correct =
          unordered_bidict<nonnegative_int, int>{
              {0_n, 3},
              {1_n, 6},
              {2_n, 5},
              {3_n, 2},
          };

      CHECK(result == correct);
    }

    SUBCASE("input has duplicates") {
      std::vector<int> input = {3, 6, 5, 3, 2};

      CHECK_THROWS(unordered_bidict_from_enumerating(input));
    }
  }

  TEST_CASE("unordered_bidict_from_enumerating(std::set<T>)") {
    std::set<std::string> input = {"a", "c", "b"};

    unordered_bidict<nonnegative_int, std::string> correct = {
        {0_n, "a"},
        {1_n, "b"},
        {2_n, "c"},
    };

    unordered_bidict<nonnegative_int, std::string> result =
        unordered_bidict_from_enumerating(input);

    CHECK(result == correct);
  }
}
