#include "utils/containers/contains_duplicates.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("contains_duplicates(T)",
                     T,
                     std::vector<int>,
                     std::unordered_multiset<int>,
                     std::multiset<int>) {
    SUBCASE("container has duplicates") {
      T input = {2, 7, 3, 4, 2, 1};

      bool result = contains_duplicates(input);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("container does not have duplicates") {
      T input = {2, 7, 3, 4, 1};

      bool result = contains_duplicates(input);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("container is empty") {
      T input = {};

      bool result = contains_duplicates(input);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
