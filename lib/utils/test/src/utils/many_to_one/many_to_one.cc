#include "utils/many_to_one/many_to_one.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/multiset.h"
#include "utils/containers/multiset_of.h"
#include "test/utils/doctest/fmt/unordered_set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ManyToOne") {
    ManyToOne<int, std::string> m;

    m.insert({1, "one"});
    m.insert({10, "one"});
    m.insert({20, "two"});
    m.insert({100, "one"});
    m.insert({2, "two"});

    SUBCASE("add redundant mapping") {
      ManyToOne<int, std::string> correct = m;

      m.insert({20, "two"});   

      ManyToOne<int, std::string> result = m;

      CHECK(result == correct);
    }

    SUBCASE("add conflicting mapping") {
      CHECK_THROWS(m.insert({20, "one"}));
    }

    SUBCASE("at_l") {
      std::string result = m.at_l(10);  

      std::string correct = "one";

      CHECK(result == correct);
    }

    SUBCASE("at_r") {
      std::unordered_set<int> result = m.at_r("two"); 

      std::unordered_set<int> correct = {2, 20};

      CHECK(result == correct);
    }

    SUBCASE("left_values") {
      std::unordered_set<int> result = m.left_values();

      std::unordered_set<int> correct = {
        1, 10, 100, 2, 20,
      };

      CHECK(result == correct);
    }

    SUBCASE("right_values") {
      std::unordered_set<std::string> result = m.right_values();      

      std::unordered_set<std::string> correct = {"one", "two"};

      CHECK(result == correct);
    }

    SUBCASE("initialization") {
      SUBCASE("basic usage") {
        ManyToOne<int, std::string> result = ManyToOne<int, std::string>{
          {{1, 10, 100}, "one"},
          {{2, 20}, "two"},
        };

        ManyToOne<int, std::string> correct = m;

        CHECK(result == m);
      } 

      SUBCASE("lhs of an initialization pair is empty") {
        CHECK_THROWS(
          ManyToOne<int, std::string>{
            {{1, 10, 100}, "one"},
            {{}, "two"},
          });
      }

      SUBCASE("initialization pairs conflict") {
        CHECK_THROWS(
          ManyToOne<int, std::string>{
            {{1, 10, 100}, "one"},
            {{2, 20, 10}, "two"},
          });
      }
    }
  }

  TEST_CASE("fmt::to_string(ManyToOne<L, R>)") {
    ManyToOne<int, std::string> input = ManyToOne<int, std::string>{
      {{1, 10, 100}, "one"}, 
      {{2, 20}, "two"}, 
    };

    std::string result = fmt::to_string(input);
    std::string correct = "{{{1, 10, 100}, one}, {{2, 20}, two}}";

    CHECK(multiset_of(result) == multiset_of(correct));
  }
}
