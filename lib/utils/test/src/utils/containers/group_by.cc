#include "utils/containers/group_by.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("group_by(std::unordered_set<V>, F)") {
    std::unordered_set<int> input = {0, 3, 2, 9, 8};

    OneToMany<int, int> result = group_by(input, [](int x) { return x % 3; });
    OneToMany<int, int> correct = {
        {0, {0, 3, 9}},
        {2, {2, 8}},
    };

    CHECK(result == correct);
  }

  TEST_CASE("group_by(std::vector<V>, F)") {
    std::vector<int> input = {0, 3, 0, 2, 2, 9, 8, 9};

    std::unordered_map<int, std::vector<int>> result =
        group_by(input, [](int x) { return x % 3; });
    std::unordered_map<int, std::vector<int>> correct = {
        {0, {0, 3, 0, 9, 9}},
        {2, {2, 2, 8}},
    };

    CHECK(result == correct);
  }

  TEST_CASE("group_by(std::set<V>, F)") {
    std::set<int> input = {0, 3, 2, 9, 8};

    OneToMany<int, int> result = group_by(input, [](int x) { return x % 3; });
    OneToMany<int, int> correct = {
        {0, {0, 3, 9}},
        {2, {2, 8}},
    };

    CHECK(result == correct);
  }
}
