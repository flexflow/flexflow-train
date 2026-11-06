#include "utils/bidict/algorithms/bidict_transform.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("bidict_transform") {
    SUBCASE("function preserves bidict structure") {
      bidict<int, std::string> dict = {
          {1, "one"},
          {2, "two"},
      };

      bidict<std::string, int> result = bidict_transform(
          dict, [](int k, std::string const &v) -> std::pair<std::string, int> {
            return std::make_pair(v, k + 1);
          });
      bidict<std::string, int> correct = {
          {"one", 2},
          {"two", 3},
      };

      CHECK(result == correct);
    }

    SUBCASE("function causes violation of bidict requirements") {
      bidict<int, std::string> dict = {
          {1, "one"},
          {2, "two"},
      };

      CHECK_THROWS(bidict_transform(
          dict, [](int k, std::string const &v) -> std::pair<std::string, int> {
            return std::make_pair(v, 3);
          }));
    }

    SUBCASE(
        "function collapses bidict, but output still has bidict structure") {
      bidict<int, std::string> dict = {
          {1, "one"},
          {2, "two"},
      };

      bidict<std::string, int> result = bidict_transform(
          dict, [](int k, std::string const &v) -> std::pair<std::string, int> {
            return std::make_pair("three", 3);
          });
      bidict<std::string, int> correct = {
          {"three", 3},
      };

      CHECK(result == correct);
    }
  }
}
