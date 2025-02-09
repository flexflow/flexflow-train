#include "utils/containers/zip3_strict.h"
#include "test/utils/doctest/fmt/tuple.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip3_strict(std::vector<A>, std::vector<B>, std::vector<C>)") {
    SUBCASE("types are same") {
      std::vector<int> input_a = {2, 1, 2};
      std::vector<int> input_b = {5, 4, 5};
      std::vector<int> input_c = {3, 4, 3};

      std::vector<std::tuple<int, int, int>> result =
          zip3_strict(input_a, input_b, input_c);
      std::vector<std::tuple<int, int, int>> correct = {
          {2, 5, 3}, {1, 4, 4}, {2, 5, 3}};

      CHECK(result == correct);
    }

    SUBCASE("types are different") {
      std::vector<int> input_a = {2, 1, 2};
      std::vector<std::string> input_b = {"a", "d", "d"};
      std::vector<std::vector<int>> input_c = {{1, 2}, {}, {3, 1}};

      std::vector<std::tuple<int, std::string, std::vector<int>>> result =
          zip3_strict(input_a, input_b, input_c);
      std::vector<std::tuple<int, std::string, std::vector<int>>> correct = {
          {2, "a", {1, 2}},
          {1, "d", {}},
          {2, "d", {3, 1}},
      };

      CHECK(result == correct);
    }

    SUBCASE("A list is shortest") {
      std::vector<int> input_a = {2};
      std::vector<int> input_b = {5, 4, 5};
      std::vector<int> input_c = {3, 4};

      CHECK_THROWS(zip3_strict(input_a, input_b, input_c));
    }

    SUBCASE("B list is shortest") {
      std::vector<int> input_a = {2, 1, 2, 4};
      std::vector<int> input_b = {5, 4};
      std::vector<int> input_c = {3, 4, 3};

      CHECK_THROWS(zip3_strict(input_a, input_b, input_c));
    }

    SUBCASE("C list is shortest") {
      std::vector<int> input_a = {2, 1, 2};
      std::vector<int> input_b = {5, 4, 5};
      std::vector<int> input_c = {3, 3};

      CHECK_THROWS(zip3_strict(input_a, input_b, input_c));
    }

    SUBCASE("one list is empty") {
      std::vector<int> input_a = {2, 1, 2};
      std::vector<int> input_b = {5, 4, 5};
      std::vector<int> input_c = {};

      CHECK_THROWS(zip3_strict(input_a, input_b, input_c));
    }

    SUBCASE("all lists are empty") {
      std::vector<int> input_a = {};
      std::vector<int> input_b = {};
      std::vector<int> input_c = {};

      std::vector<std::tuple<int, int, int>> result =
          zip3_strict(input_a, input_b, input_c);
      std::vector<std::tuple<int, int, int>> correct = {};

      CHECK(result == correct);
    }
  }
}
