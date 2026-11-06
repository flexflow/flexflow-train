#include "utils/unordered_bidict/algorithms/unordered_bidict_from_unstructured_relation.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_from_unstructured_relation") {
    SUBCASE("relation is one-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {2, "two"},
      };

      unordered_bidict<int, std::string> result =
          unordered_bidict_from_unstructured_relation(input);
      unordered_bidict<int, std::string> correct = {
          {1, "one"},
          {2, "two"},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is one-to-many") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ONE"},
          {2, "two"},
      };

      CHECK_THROWS(unordered_bidict_from_unstructured_relation(input));
    }

    SUBCASE("relation is many-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "odd"},
          {2, "even"},
          {3, "odd"},
      };

      CHECK_THROWS(unordered_bidict_from_unstructured_relation(input));
    }

    SUBCASE("relation is none of the above") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "odd"},
          {1, "ODD"},
          {2, "even"},
          {3, "odd"},
      };

      CHECK_THROWS(unordered_bidict_from_unstructured_relation(input));
    }
  }
}
