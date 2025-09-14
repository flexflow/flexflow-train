#include "utils/containers/require_all_same1.h"
#include "test/utils/doctest/fmt/expected.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "test/utils/doctest/fmt/optional.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/unordered_multiset.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "utils/expected.h"
#include <doctest/doctest.h>
#include <optional>
#include <set>
#include <unordered_set>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE_TEMPLATE("require_all_same1(T)",
                     T,
                     std::vector<int>,
                     std::unordered_set<int>,
                     std::unordered_multiset<int>,
                     std::set<int>,
                     std::multiset<int>) {
    SUBCASE("input is empty") {
      T input = {};

      CHECK_THROWS(require_all_same1(input));
    }

    SUBCASE("input elements are all the same") {
      T input = {1, 1, 1};

      int result = require_all_same1(input);
      int correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("input elements are not all the same") {
      T input = {1, 1, 2, 1};

      CHECK_THROWS(require_all_same1(input));
    }
  }
}
