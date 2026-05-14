#include "utils/graph/query_set.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("query_set") {
    SUBCASE("handles optional values correctly") {
      std::optional<int> nopt = std::nullopt;
      std::optional<int> three = 3;
      std::optional<int> five = 5;

      query_set<std::optional<int>> q1 =
          query_set<std::optional<int>>::matchall();

      query_set<std::optional<int>> q2 =
          query_set<std::optional<int>>::match_values_in(std::set{
              nopt,
              three,
          });

      query_set<std::optional<int>> q3 =
          query_set<std::optional<int>>::match_single_value(nopt);

      query_set<std::optional<int>> q4 =
          query_set<std::optional<int>>::match_none();

      CHECK(includes(q1, nopt));
      CHECK(includes(q1, three));
      CHECK(includes(q1, five));

      CHECK(includes(q2, nopt));
      CHECK(includes(q2, three));
      CHECK_FALSE(includes(q2, five));

      CHECK(includes(q3, nopt));
      CHECK_FALSE(includes(q3, three));
      CHECK_FALSE(includes(q3, five));

      CHECK_FALSE(includes(q4, nopt));
      CHECK_FALSE(includes(q4, three));
      CHECK_FALSE(includes(q4, five));
    }
  }
}
