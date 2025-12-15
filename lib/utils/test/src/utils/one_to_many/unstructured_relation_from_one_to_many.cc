#include <doctest/doctest.h>
#include "utils/one_to_many/unstructured_relation_from_one_to_many.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/pair.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("unstructured_relation_from_one_to_many") {
    OneToMany<int, std::string> input = {
      {1, {"one", "ONE"}},
      {2, {"two"}},
    };

    std::unordered_set<std::pair<int, std::string>> result = unstructured_relation_from_one_to_many(input);
    std::unordered_set<std::pair<int, std::string>> correct = {
      {1, "one"},
      {1, "ONE"},
      {2, "two"},
    };

    CHECK(result == correct);
  }
}
