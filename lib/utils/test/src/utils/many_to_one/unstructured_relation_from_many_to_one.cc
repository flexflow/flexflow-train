#include <doctest/doctest.h>
#include "utils/many_to_one/unstructured_relation_from_many_to_one.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unstructured_relation_from_many_to_one") {
    ManyToOne<int, std::string> input = {
      {{1, 3}, "odd"},
      {{2}, "even"},
    };

    std::unordered_set<std::pair<int, std::string>> result = unstructured_relation_from_many_to_one(input);
    std::unordered_set<std::pair<int, std::string>> correct = {
      {1, "odd"},
      {2, "even"},
      {3, "odd"},
    };

    CHECK(result == correct);
  }
}
