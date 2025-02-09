#include "utils/containers/zip_with_strict.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>
#include <stdexcept>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip_with_strict(std::vector<T1>, std::vector<T2>, F)") {
    SUBCASE("result types and input types are all different") {
      std::vector<int> v1 = {1, 3, 4, 3};
      std::vector<std::string> v2 = {"aa", "cc", "bb", "dd"};

      std::vector<std::pair<int, std::string>> result =
          zip_with(v1, v2, [](int x1, std::string const &x2) {
            return std::make_pair(x1, x2);
          });
      std::vector<std::pair<int, std::string>> correct = {
          {1, "aa"},
          {3, "cc"},
          {4, "bb"},
          {3, "dd"},
      };

      CHECK(result == correct);
    }

    SUBCASE("input lengths don't match") {
      auto add = [](int x1, int x2) { return x1 + x2; };

      std::vector<int> shorter = {1, 2};
      std::vector<int> longer = {1, 3, 5, 7};

      SUBCASE("first input is shorter") {
        CHECK_THROWS(zip_with_strict(shorter, longer, add));
      }

      SUBCASE("second input is shorter") {
        CHECK_THROWS(zip_with_strict(longer, shorter, add));
      }
    }

    SUBCASE("properly handles empty inputs") {
      std::vector<int> empty = {};

      auto throw_err = [](int x1, int x2) -> int {
        throw std::runtime_error("error");
      };

      std::vector<int> result = zip_with(empty, empty, throw_err);
      std::vector<int> correct = empty;

      CHECK(result == correct);
    }
  }
}
