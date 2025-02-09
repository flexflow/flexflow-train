#include "utils/containers/zip_strict.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip_strict(std::vector<L>, std::vector<R>)") {
    SUBCASE("input lengths are the same") {
      std::vector<std::string> lhs = {"a", "b", "b"};
      std::vector<int> rhs = {5, 4, 8};

      std::vector<std::pair<std::string, int>> result = zip_strict(lhs, rhs);
      std::vector<std::pair<std::string, int>> correct = {
          {"a", 5}, {"b", 4}, {"b", 8}};

      CHECK(result == correct);
    }

    SUBCASE("input lengths are not the same") {
      std::vector<std::string> lhs = {"a", "b", "b"};
      std::vector<int> rhs = {5, 4};

      CHECK_THROWS(zip_strict(lhs, rhs));
    }
  }
}
