#include "utils/containers/permute_with_key.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck/doctest.h"
#include "utils/containers/get_all_permutations.h"
#include "utils/containers/range.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/hash/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("permute_with_key") {
    SUBCASE("generates all permutations within period") {
      std::vector<std::string> input = std::vector<std::string>{
          "alpha",
          "beta",
          "gamma",
          "delta",
      };
      int max_permutations = 4 * 3 * 2 * 1;

      std::unordered_set<std::vector<std::string>> generated_permutations =
          unordered_set_of(transform(range(max_permutations), [&](int key) {
            return permute_with_key(key, input);
          }));
      std::unordered_set<std::vector<std::string>> all_permutations =
          unordered_set_of(get_all_permutations(input));

      CHECK(generated_permutations == all_permutations);
    }

    RC_SUBCASE(
        "generates valid permutations", [](int key, std::vector<int> input) {
          std::vector<int> permuted = permute_with_key(key, input);
          return std::is_permutation(
              input.cbegin(), input.cend(), permuted.cbegin(), permuted.cend());
        });
  }
}
