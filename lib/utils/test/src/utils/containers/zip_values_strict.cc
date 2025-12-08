#include <doctest/doctest.h>
#include <string>
#include "utils/containers/zip_values_strict.h"
#include "test/utils/doctest/fmt/unordered_map.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip_values_strict") {
    SUBCASE("key sets are the same") {
      std::unordered_map<int, std::string> m1 = {
        {2, "two"},
        {3, "three"},
      };
      std::unordered_map<int, std::string> m2 = {
        {2, "TWO"},
        {3, "THREE"},
      };

      std::unordered_map<int, std::pair<std::string, std::string>>
        result = zip_values_strict(m1, m2);
      std::unordered_map<int, std::pair<std::string, std::string>>
        correct = {
        {2, {"two", "TWO"}},
        {3, {"three", "THREE"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("key sets are different but same size") {
      std::unordered_map<int, std::string> m1 = {
        {2, "two"},
        {3, "three"},
      };
      std::unordered_map<int, std::string> m2 = {
        {2, "TWO"},
        {4, "FOUR"},
      };

      CHECK_THROWS(zip_values_strict(m1, m2));
    }

    SUBCASE("key sets are subset") {
      std::unordered_map<int, std::string> m1 = {
        {2, "two"},
        {3, "three"},
      };
      std::unordered_map<int, std::string> m2 = {
        {2, "TWO"},
      };

      CHECK_THROWS(zip_values_strict(m1, m2));
    }
  }
}
