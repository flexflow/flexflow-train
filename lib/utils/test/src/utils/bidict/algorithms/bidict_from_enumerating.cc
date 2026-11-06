#include "utils/bidict/algorithms/bidict_from_enumerating.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/bidict/algorithms/bidict_left_entries.h"
#include "utils/bidict/algorithms/bidict_right_entries.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict_from_enumerating(std::vector<T>)") {
    SUBCASE("input has no duplicates") {
      std::vector<int> input = {3, 6, 5, 2};

      bidict<nonnegative_int, int> result = bidict_from_enumerating(input);

      bidict<nonnegative_int, int> correct = bidict<nonnegative_int, int>{
          {0_n, 3},
          {1_n, 6},
          {2_n, 5},
          {3_n, 2},
      };

      CHECK(result == correct);
    }

    SUBCASE("input has duplicates") {
      std::vector<int> input = {3, 6, 5, 3, 2};

      CHECK_THROWS(bidict_from_enumerating(input));
    }
  }

  TEST_CASE("bidict_from_enumerating(std::set<T>)") {
    std::set<std::string> input = {"a", "c", "b"};

    bidict<nonnegative_int, std::string> correct = {
        {0_n, "a"},
        {1_n, "b"},
        {2_n, "c"},
    };

    bidict<nonnegative_int, std::string> result =
        bidict_from_enumerating(input);

    CHECK(result == correct);
  }
}
