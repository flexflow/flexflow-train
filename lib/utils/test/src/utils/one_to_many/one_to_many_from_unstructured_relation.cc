#include "utils/one_to_many/one_to_many_from_unstructured_relation.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("one_to_many_from_unstructured_relation") {
    SUBCASE("relation is one-to-many") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ONE"},
          {2, "two"},
      };

      OneToMany<int, std::string> result =
          one_to_many_from_unstructured_relation(input);
      OneToMany<int, std::string> correct = {
          {1, {"one", "ONE"}},
          {2, {"two"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is one-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {2, "two"},
      };

      OneToMany<int, std::string> result =
          one_to_many_from_unstructured_relation(input);
      OneToMany<int, std::string> correct = {
          {1, {"one"}},
          {2, {"two"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is not one-to-many") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ONE"},
          {2, "two"},
          {3, "ONE"},
      };

      CHECK_THROWS(one_to_many_from_unstructured_relation(input));
    }
  }
}
