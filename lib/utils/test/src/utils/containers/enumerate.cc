#include "utils/containers/enumerate.h"
#include "test/utils/doctest/fmt/map.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/containers/keys.h"
#include "utils/containers/unordered_multiset_of.h"
#include "utils/containers/values.h"
#include "utils/containers/vector_of.h"
#include <doctest/doctest.h>
#include <string>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("enumerate(std::vector<T>)") {
    std::vector<std::string> input = {"zero", "one", "two", "three"};

    std::map<nonnegative_int, std::string> correct = {
        {0_n, "zero"},
        {1_n, "one"},
        {2_n, "two"},
        {3_n, "three"},
    };

    std::map<nonnegative_int, std::string> result = enumerate(input);

    CHECK(result == correct);

    SUBCASE("check iteration order") {
      std::vector<std::pair<nonnegative_int const, std::string>>
          iterated_result = vector_of(result);
      std::vector<std::pair<nonnegative_int const, std::string>>
          correct_iteration_order = {
              {0_n, "zero"},
              {1_n, "one"},
              {2_n, "two"},
              {3_n, "three"},
          };

      CHECK(iterated_result == correct_iteration_order);
    }
  }

  TEST_CASE("enumerate(std::unordered_set<T>)") {
    std::unordered_set<std::string> input = {"A", "B", "C", "D"};

    std::unordered_set<nonnegative_int> correct_keys = {0_n, 1_n, 2_n, 3_n};
    std::unordered_multiset<std::string> correct_values = {"A", "B", "C", "D"};
    std::map<nonnegative_int, std::string> result = enumerate(input);

    CHECK(keys(result) == correct_keys);
    CHECK(unordered_multiset_of(values(result)) == correct_values);
  }
}
