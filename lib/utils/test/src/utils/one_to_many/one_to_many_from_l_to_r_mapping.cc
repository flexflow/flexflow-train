#include <doctest/doctest.h>
#include "utils/one_to_many/one_to_many_from_l_to_r_mapping.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("one_to_many_from_l_to_r_mapping") {
    SUBCASE("basic usage") {
      OneToMany<int, std::string> result =
          one_to_many_from_l_to_r_mapping<int, std::string>(
              {{1, {"hello", "world"}}, {3, {"HELLO"}}});

      OneToMany<int, std::string> correct = OneToMany<int, std::string>{
        {1, {"hello", "world"}},
        {3, {"HELLO"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("throws if a value set is empty") {
      CHECK_THROWS(one_to_many_from_l_to_r_mapping<int, std::string>(
              {{1, {"hello", "world"}}, {2, {}}, {3, {"HELLO"}}}));
    }
  }
}
