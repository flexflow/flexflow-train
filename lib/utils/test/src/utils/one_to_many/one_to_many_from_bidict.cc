#include "utils/one_to_many/one_to_many_from_bidict.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("one_to_many_from_bidict(bidict<L, R>)") {
    SUBCASE("input is empty") {
      bidict<int, std::string> input = {};

      OneToMany<int, std::string> result = one_to_many_from_bidict(input); 
      OneToMany<int, std::string> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("input is nonempty") {
      bidict<int, std::string> input = {
        {1, "one"},
        {2, "two"},
      };

      OneToMany<int, std::string> result = one_to_many_from_bidict(input); 
      OneToMany<int, std::string> correct = {
        {1, {"one"}},
        {2, {"two"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("key and value types are the same") {
      bidict<int, int> input = {
        {1, -1},
        {2, -2},
      };

      OneToMany<int, int> result = one_to_many_from_bidict(input); 
      OneToMany<int, int> correct = {
        {1, {-1}},
        {2, {-2}},
      };

      CHECK(result == correct);
    }
  }
}
