#include "utils/one_to_many/one_to_many.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "utils/containers/multiset_of.h"
#include "utils/one_to_many/one_to_many_from_l_to_r_mapping.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/unordered_set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("OneToMany") {
    OneToMany<int, std::string> m;

    m.insert({1, "one"});
    m.insert({2, "two"});
    m.insert({1, "One"});
    m.insert({1, "ONE"});

    SUBCASE("inserting redundant mapping") {
      OneToMany<int, std::string> correct = m;

      m.insert({1, "one"});

      OneToMany<int, std::string> result = m;

      CHECK(result == correct);
    }

    SUBCASE("inserting conflicting mapping") {
      CHECK_THROWS(m.insert({2, "one"}));
    }

    SUBCASE("at_l") {
      std::unordered_set<std::string> result = m.at_l(1); 

      std::unordered_set<std::string> correct = {"one", "One", "ONE"};

      CHECK(result == correct);
    }

    SUBCASE("at_r") {
      int result = m.at_r("One");

      int correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("left_values") {
      std::unordered_set<int> result = m.left_values();  

      std::unordered_set<int> correct = {1, 2};

      CHECK(result == correct);
    }

    SUBCASE("right_values") {
      std::unordered_set<std::string> result = m.right_values();  

      std::unordered_set<std::string> correct = {"one", "One", "ONE", "two"};

      CHECK(result == correct);
    }

    SUBCASE("initialization") {
      SUBCASE("basic usage") {
        OneToMany<int, std::string> result = OneToMany<int, std::string>{
          {1, {"one", "One", "ONE"}}, 
          {2, {"two"}},
        };

        OneToMany<int, std::string> correct = m;

        CHECK(result == correct);
      }

      SUBCASE("rhs of an initialization pair is empty") {
        CHECK_THROWS(
          OneToMany<int, std::string>{
            {1, {}}, 
            {2, {"two"}},
          });
      }

      SUBCASE("conflicting values") {
        CHECK_THROWS(
          OneToMany<int, std::string>{
            {1, {"two"}}, 
            {2, {"two"}},
          });
      }
    }
  }

  TEST_CASE("fmt::to_string(OneToMany<L, R>)") {
    OneToMany<int, std::string> input =
        one_to_many_from_l_to_r_mapping<int, std::string>(
            {{1, {"hello", "world"}}, {2, {}}, {3, {"HELLO"}}});

    std::string result = fmt::to_string(input);
    std::string correct = "{{1, {hello, world}}, {3, {HELLO}}}";

    CHECK(multiset_of(result) == multiset_of(correct));
  }
}
