#include "op-attrs/ff_ordered/zip_with.h"
#include "test/utils/doctest/fmt/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("zip_with(FFOrdered<T1>, FFOrdered<T2>, F)") {
    SUBCASE("result types and input types are all different") {
      FFOrdered<int> v1 = FFOrdered<int>{1, 3, 4, 3};
      FFOrdered<std::string> v2 =
          FFOrdered<std::string>{"aa", "cc", "bb", "dd"};

      FFOrdered<std::pair<int, std::string>> result =
          zip_with(v1, v2, [](int x1, std::string const &x2) {
            return std::make_pair(x1, x2);
          });
      FFOrdered<std::pair<int, std::string>> correct =
          FFOrdered<std::pair<int, std::string>>{
              {1, "aa"},
              {3, "cc"},
              {4, "bb"},
              {3, "dd"},
          };

      CHECK(result == correct);
    }

    SUBCASE("input lengths don't match") {
      auto add = [](int x1, int x2) { return x1 + x2; };

      FFOrdered<int> shorter = FFOrdered<int>{1, 2};
      FFOrdered<int> longer = FFOrdered<int>{1, 3, 5, 7};

      SUBCASE("first input is shorter") {
        FFOrdered<int> result = zip_with(shorter, longer, add);
        FFOrdered<int> correct = FFOrdered<int>{1 + 1, 2 + 3};

        CHECK(result == correct);
      }

      SUBCASE("second input is shorter") {
        FFOrdered<int> result = zip_with(longer, shorter, add);
        FFOrdered<int> correct = FFOrdered<int>{1 + 1, 2 + 3};

        CHECK(result == correct);
      }
    }

    SUBCASE("properly handles empty inputs") {
      FFOrdered<int> nonempty = FFOrdered<int>{1, 2};
      FFOrdered<int> empty = {};

      auto throw_err = [](int x1, int x2) -> int {
        throw std::runtime_error("error");
      };

      SUBCASE("first input is empty") {
        FFOrdered<int> result = zip_with(empty, nonempty, throw_err);
        FFOrdered<int> correct = empty;

        CHECK(result == correct);
      }

      SUBCASE("second input is empty") {
        FFOrdered<int> result = zip_with(nonempty, empty, throw_err);
        FFOrdered<int> correct = empty;

        CHECK(result == correct);
      }

      SUBCASE("both inputs are empty") {
        FFOrdered<int> result = zip_with(empty, empty, throw_err);
        FFOrdered<int> correct = empty;

        CHECK(result == correct);
      }
    }
  }
}
