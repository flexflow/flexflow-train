#include "utils/one_to_many/one_to_many.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "test/utils/doctest/fmt/set.h"
#include "utils/containers/multiset_of.h"
#include "utils/one_to_many/one_to_many_from_l_to_r_mapping.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include <doctest/doctest.h>

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
      nonempty_set<std::string> result = m.at_l(1);

      nonempty_set<std::string> correct = {"one", "One", "ONE"};

      CHECK(result == correct);
    }

    SUBCASE("at_r") {
      int result = m.at_r("One");

      int correct = 1;

      CHECK(result == correct);
    }

    SUBCASE("left_values") {
      std::set<int> result = m.left_values();

      std::set<int> correct = {1, 2};

      CHECK(result == correct);
    }

    SUBCASE("right_values") {
      std::set<std::string> result = m.right_values();

      std::set<std::string> correct = {"one", "One", "ONE", "two"};

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
        CHECK_THROWS(OneToMany<int, std::string>{
            {1, {}},
            {2, {"two"}},
        });
      }

      SUBCASE("conflicting values") {
        CHECK_THROWS(OneToMany<int, std::string>{
            {1, {"two"}},
            {2, {"two"}},
        });
      }
    }
  }

  TEST_CASE("adl_serializer<OneToMany<L, R>>") {
    OneToMany<int, std::string> deserialized = OneToMany<int, std::string>{
        {2, {"two", "TWO"}},
        {3, {"three"}},
        {4, {"four"}},
    };

    nlohmann::json serialized = std::set<std::pair<int, std::string>>{
        {2, "two"},
        {2, "TWO"},
        {3, "three"},
        {4, "four"},
    };

    SUBCASE("to_json") {
      nlohmann::json result = deserialized;
      nlohmann::json correct = serialized;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      OneToMany<int, std::string> result = serialized;
      OneToMany<int, std::string> correct = deserialized;

      CHECK(result == correct);
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

  TEST_CASE("unstructured_relation_from_one_to_many") {
    OneToMany<int, std::string> input = {
        {1, {"one", "ONE"}},
        {2, {"two"}},
    };

    std::unordered_set<std::pair<int, std::string>> result =
        unstructured_relation_from_one_to_many(input);
    std::unordered_set<std::pair<int, std::string>> correct = {
        {1, "one"},
        {1, "ONE"},
        {2, "two"},
    };

    CHECK(result == correct);
  }

  TEST_CASE("one_to_many_from_unstructured_relation") {
    SUBCASE("relation is one-to-many") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ONE"},
          {2, "two"},
      };

      OneToMany<int, std::string> result =
          one_to_many_from_unstructured_relation(input);
      OneToMany<int, std::string> correct = {
          {1, {"one", "ONE"}},
          {2, {"two"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is one-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {2, "two"},
      };

      OneToMany<int, std::string> result =
          one_to_many_from_unstructured_relation(input);
      OneToMany<int, std::string> correct = {
          {1, {"one"}},
          {2, {"two"}},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is not one-to-many") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ONE"},
          {2, "two"},
          {3, "ONE"},
      };

      CHECK_THROWS(one_to_many_from_unstructured_relation(input));
    }
  }
}
