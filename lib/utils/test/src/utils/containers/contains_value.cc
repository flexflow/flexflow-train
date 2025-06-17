#include <doctest/doctest.h>
#include "utils/containers/contains_value.h"
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("contains_value(std::unordered_map<K, V>, V)") {
    std::unordered_map<int, std::string> m = {
      {1, "one"},
      {3, "three"},
      {4, "three"},
    };

    SUBCASE("returns true if the value is in the map") {
      bool result = contains_value(m, std::string{"one"});
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("returns false if the value is not in the map") {
      bool result = contains_value(m, std::string{"two"});
      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("contains_value(std::map<K, V>, V)") {
    std::map<int, std::string> m = {
      {1, "one"},
      {3, "three"},
      {4, "three"},
    };

    SUBCASE("returns true if the value is in the map") {
      bool result = contains_value(m, std::string{"one"});
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("returns false if the value is not in the map") {
      bool result = contains_value(m, std::string{"two"});
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
