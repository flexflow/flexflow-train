#include "utils/unordered_bidict/algorithms/unordered_bidict_from_map.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unordered_bidict_from_map(std::map<L, R>)") {
    SUBCASE("map values do not contain duplicates") {
      std::map<int, std::string> input = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      unordered_bidict<int, std::string> result =
          unordered_bidict_from_map(input);
      unordered_bidict<int, std::string> correct = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      CHECK(result == correct);
    }

    SUBCASE("map values contain duplicates") {
      std::map<int, std::string> input = {
          {1, "odd"},
          {2, "even"},
          {3, "odd"},
      };

      CHECK_THROWS(unordered_bidict_from_map(input));
    }
  }

  TEST_CASE("unordered_bidict_from_map(std::unordered_map<L, R>)") {
    SUBCASE("map values do not contain duplicates") {
      std::unordered_map<int, std::string> input = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      unordered_bidict<int, std::string> result =
          unordered_bidict_from_map(input);
      unordered_bidict<int, std::string> correct = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      CHECK(result == correct);
    }

    SUBCASE("map values contain duplicates") {
      std::unordered_map<int, std::string> input = {
          {1, "odd"},
          {2, "even"},
          {3, "odd"},
      };

      CHECK_THROWS(unordered_bidict_from_map(input));
    }
  }
}
