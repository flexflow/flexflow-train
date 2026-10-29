#include "utils/nonnegative_int/nonnegative_range_inclusive.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("nonnegative_range_inclusive") {
    SUBCASE("start < end") {
      nonnegative_int start = 2_n;
      nonnegative_int end = 4_n;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end);
      std::vector<nonnegative_int> correct = {2_n, 3_n, 4_n};

      CHECK(result == correct);
    }

    SUBCASE("start == end") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 4_n;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end);
      std::vector<nonnegative_int> correct = {4_n};

      CHECK(result == correct);
    }

    SUBCASE("start > end") {
      nonnegative_int start = 6_n;
      nonnegative_int end = 4_n;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end);
      std::vector<nonnegative_int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < end, step > 1, range evenly divisibly by step") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 10_n;
      int step = 2;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {4_n, 6_n, 8_n, 10_n};

      CHECK(result == correct);
    }

    SUBCASE("start < end, step > 1, range not evenly divisibly by step") {
      nonnegative_int start = 5_n;
      nonnegative_int end = 10_n;
      int step = 2;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {5_n, 7_n, 9_n};

      CHECK(result == correct);
    }

    SUBCASE("start < end, step == 0") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 10_n;
      int step = 0;

      CHECK_THROWS(nonnegative_range_inclusive(start, end, step));
    }

    SUBCASE("start < end, step < 0") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 10_n;
      int step = -1;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start == end, step > 1") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 4_n;
      int step = -2;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {4_n};

      CHECK(result == correct);
    }

    SUBCASE("start == end, step == 0") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 4_n;
      int step = 0;

      CHECK_THROWS(nonnegative_range_inclusive(start, end, step));
    }

    SUBCASE("start == end, step < 0") {
      nonnegative_int start = 4_n;
      nonnegative_int end = 4_n;
      int step = -2;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {4_n};

      CHECK(result == correct);
    }

    SUBCASE("start > end, step > 1") {
      nonnegative_int start = 12_n;
      nonnegative_int end = 4_n;
      int step = 2;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start > end, step == 0") {
      nonnegative_int start = 12_n;
      nonnegative_int end = 4_n;
      int step = 0;

      CHECK_THROWS(nonnegative_range_inclusive(start, end, step));
    }

    SUBCASE("start > end, step < 0, range evenly divisible by step") {
      nonnegative_int start = 12_n;
      nonnegative_int end = 4_n;
      int step = -4;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {12_n, 8_n, 4_n};

      CHECK(result == correct);
    }

    SUBCASE("start > end, step < 0, range not evenly divisible by step") {
      nonnegative_int start = 14_n;
      nonnegative_int end = 4_n;
      int step = -3;

      std::vector<nonnegative_int> result =
          nonnegative_range_inclusive(start, end, step);
      std::vector<nonnegative_int> correct = {14_n, 11_n, 8_n, 5_n};

      CHECK(result == correct);
    }
  }
}
