#include "utils/containers/compute_mean.h"
#include <doctest/doctest.h>
#include <vector>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compute_mean") {
    SUBCASE("input is empty") {
      std::vector<float> input = {};

      CHECK_THROWS(compute_mean(input));
    }

    SUBCASE("input has one element") {
      std::vector<float> input = {3.8f};

      float result = compute_mean(input);
      float correct = 3.8f;

      CHECK(result == correct);
    }

    SUBCASE("input has all the same element") {
      std::vector<float> input = {2.0f, 2.0f, 2.0f, 2.0f, 2.0f};

      float result = compute_mean(input);
      float correct = 2.0f;

      CHECK(result == correct);
    }

    SUBCASE("input_has_different elements") {
      std::vector<float> input = {1.0f, 5.0f, 3.0f};

      float result = compute_mean(input);
      float correct = 3.0f;

      CHECK(result == correct);
    }

    SUBCASE("input takes repeats into account") {
      std::vector<float> input = {1.0f, 5.0f, 3.0f, 5.0f};

      float result = compute_mean(input);
      float correct = (1.0f + 5.0f + 3.0f + 5.0f) / 4;

      CHECK(result == correct);
    }
  }
}
