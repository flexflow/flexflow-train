#include "utils/containers/slice.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <vector>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("slice") {
    std::vector<int> v = {1, 2, 3, 4, 5};

    SUBCASE("Basic slice") {
      auto result = slice(v, 1, 4);
      std::vector<int> correct = {2, 3, 4};
      CHECK(result == correct);
    }

    SUBCASE("From beginning to index") {
      auto result = slice(v, 0, 3);
      std::vector<int> correct = {1, 2, 3};
      CHECK(result == correct);
    }

    SUBCASE("From index to end") {
      auto result = slice(v, 2, std::nullopt);
      std::vector<int> correct = {3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("All of the vector") {
      auto result = slice(v, 0, std::nullopt);
      std::vector<int> correct = {1, 2, 3, 4, 5};
      CHECK(result == correct);
    }

    SUBCASE("Start greater than end") {
      auto result = slice(v, 3, 1);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("Start equal to end") {
      auto result = slice(v, 3, 3);
      std::vector<int> correct = {};
      CHECK(result == correct);
    }

    SUBCASE("Negative indices") {
      auto result = slice(v, -3, -1);
      std::vector<int> correct = {3, 4};
      CHECK(result == correct);
    }

    SUBCASE("Upper index is out of bounds by 1") {
      CHECK_THROWS(slice(v, 2, 6));
    }

    SUBCASE("Lower index is out of bounds by 1") {
      CHECK_THROWS(slice(v, -6, 2));
    }
  }
}
