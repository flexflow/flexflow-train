#include "utils/many_to_one/many_to_one_from_unstructured_relation.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("many_to_one_from_unstructured_relation") {
    SUBCASE("relation is many-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "odd"},
          {2, "even"},
          {3, "odd"},
      };

      ManyToOne<int, std::string> result =
          many_to_one_from_unstructured_relation(input);
      ManyToOne<int, std::string> correct = {
          {{1, 3}, "odd"},
          {{2}, "even"},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is one-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      ManyToOne<int, std::string> result =
          many_to_one_from_unstructured_relation(input);
      ManyToOne<int, std::string> correct = {
          {{1}, "one"},
          {{2}, "two"},
          {{3}, "three"},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is not many-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ODD"},
          {2, "two"},
          {3, "ODD"},
      };

      CHECK_THROWS(many_to_one_from_unstructured_relation(input));
    }
  }
}
