#include "utils/containers/range_inclusive.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("range_inclusive") {
    SUBCASE("end > 0") {
      int end = 4;

      std::vector<int> result = range_inclusive(end);
      std::vector<int> correct = {0, 1, 2, 3, 4};

      CHECK(result == correct);
    }

    SUBCASE("end == 0") {
      int end = 0;

      std::vector<int> result = range_inclusive(end);
      std::vector<int> correct = {0};

      CHECK(result == correct);
    }

    SUBCASE("end < 0") {
      int end = -2;

      std::vector<int> result = range_inclusive(end);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < end") {
      int start = 2;
      int end = 4;

      std::vector<int> result = range_inclusive(start, end);
      std::vector<int> correct = {2, 3, 4};

      CHECK(result == correct);
    }

    SUBCASE("start == end") {
      int start = 4;
      int end = 4;

      std::vector<int> result = range_inclusive(start, end);
      std::vector<int> correct = {4};

      CHECK(result == correct);
    }

    SUBCASE("start > end") {
      int start = 6;
      int end = 4;

      std::vector<int> result = range_inclusive(start, end);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < end, step > 1, range evenly divisibly by step") {
      int start = 4;
      int end = 10;
      int step = 2;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {4, 6, 8, 10};

      CHECK(result == correct);
    }

    SUBCASE("start < end, step > 1, range not evenly divisibly by step") {
      int start = 5;
      int end = 10;
      int step = 2;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {5, 7, 9};

      CHECK(result == correct);
    }

    SUBCASE("start < end, step == 0") {
      int start = 4;
      int end = 10;
      int step = 0;

      CHECK_THROWS(range_inclusive(start, end, step));
    }

    SUBCASE("start < end, step < 0") {
      int start = 4;
      int end = 10;
      int step = -1;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start == end, step > 1") {
      int start = 4;
      int end = 4;
      int step = -2;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {4};

      CHECK(result == correct);
    }

    SUBCASE("start == end, step == 0") {
      int start = 4;
      int end = 4;
      int step = 0;

      CHECK_THROWS(range_inclusive(start, end, step));
    }

    SUBCASE("start == end, step < 0") {
      int start = 4;
      int end = 4;
      int step = -2;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {4};

      CHECK(result == correct);
    }

    SUBCASE("start > end, step > 1") {
      int start = 12;
      int end = 4;
      int step = 2;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start > end, step == 0") {
      int start = 12;
      int end = 4;
      int step = 0;

      CHECK_THROWS(range_inclusive(start, end, step));
    }

    SUBCASE("start > end, step < 0, range evenly divisible by step") {
      int start = 12;
      int end = 4;
      int step = -4;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {12, 8, 4};

      CHECK(result == correct);
    }

    SUBCASE("start > end, step < 0, range not evenly divisible by step") {
      int start = 14;
      int end = 4;
      int step = -3;

      std::vector<int> result = range_inclusive(start, end, step);
      std::vector<int> correct = {14, 11, 8, 5};

      CHECK(result == correct);
    }
  }
}
