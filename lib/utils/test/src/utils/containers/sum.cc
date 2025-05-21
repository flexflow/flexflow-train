#include "utils/containers/sum.h"
#include <doctest/doctest.h>
#include <vector>
#include "utils/positive_int/positive_int.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("sum(std::vector<int>)") {
    SUBCASE("input is empty") {
      std::vector<int> input = {};

      int result = sum(input);
      int correct = 0;

      CHECK(result == correct);
    }

    SUBCASE("input is not empty") {
      std::vector<int> input = {1, 3, 2};

      int result = sum(input);
      int correct = 6;

      CHECK(result == correct);
    }
  }

  TEST_CASE("sum(std::vector<positive_int>)") {
    SUBCASE("returns the sum if the input is not empty") {
      std::vector<positive_int> input = {3_p, 9_p, 3_p}; 

      positive_int result = sum(input);
      positive_int correct = 15_p;

      CHECK(result == correct);
    } 

    SUBCASE("throws an error if the input is empty, as then 0 should be returned") {
      std::vector<positive_int> input = {}; 

      CHECK_THROWS(sum(input));
    }
  }
}
