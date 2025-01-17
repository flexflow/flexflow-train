#include "utils/containers/intersection.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("intersection(S, S)", S, std::unordered_set<int>, std::set<int>) {
    S input_l = {1, 2, 3};
    S input_r = {2, 3, 5};

    S result = intersection(input_l, input_r);
    S correct = {2, 3};

    CHECK(result == correct);
  }

  TEST_CASE_TEMPLATE("intersection(C<S>)", S, std::unordered_set<int>, std::set<int>) {
    SUBCASE("input is empty container") {
      std::vector<S> input = {};

      std::optional<S> result = intersection(input);
      std::optional<S> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("input is has only one set") {
      std::vector<S> input = {{1, 2, 3}};

      std::optional<S> result = intersection(input);
      std::optional<S> correct = {{1, 2, 3}};

      CHECK(result == correct);
    }

    SUBCASE("input has multiple sets") {
      std::vector<S> input = {
          {1, 2, 3}, {2, 3, 4}, {3, 4, 5}};

      std::optional<S> result = intersection(input);
      std::optional<S> correct = {{3}};

      CHECK(result == correct);
    }
  }
}
