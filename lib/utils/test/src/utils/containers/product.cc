#include "utils/containers/product.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <climits>
#include <doctest/doctest.h>
#include <set>
#include <unordered_set>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE_TEMPLATE("product",
                     C,
                     std::vector<int>,
                     std::vector<double>,
                     std::set<int>,
                     std::unordered_set<int>) {

    SUBCASE("non-empty container") {
      C input = {1, -2, 3, 5};
      auto correct = -30;
      auto result = product(input);
      CHECK(correct == result);
    }

    SUBCASE("empty container") {
      C input = {};
      auto correct = 1;
      auto result = product(input);
      CHECK(correct == result);
    }
  }

  TEST_CASE("product(std::vector<nonnegative_int>)") {
    SUBCASE("non-empty container") {
      std::vector<nonnegative_int> input = {1_n, 2_n, 3_n, 5_n};
      nonnegative_int correct = 30_n;
      auto result = product(input);
      CHECK(correct == result);
    }

    SUBCASE("empty container") {
      std::vector<nonnegative_int> input = {5_n};
      nonnegative_int correct = 5_n;
      // correct = nonnegative_int{x};
      // CHECK(x == 3);
      nonnegative_int result = product(input);
      CHECK(correct == correct);
    }
  }
}
