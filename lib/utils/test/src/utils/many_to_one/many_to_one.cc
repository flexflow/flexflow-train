#include "utils/many_to_one/many_to_one.h"
#include "test/utils/doctest/fmt/multiset.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/containers/multiset_of.h"
#include <doctest/doctest.h>

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
          1,
          10,
          100,
          2,
          20,
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
        CHECK_THROWS(ManyToOne<int, std::string>{
            {{1, 10, 100}, "one"},
            {{}, "two"},
        });
      }

      SUBCASE("initialization pairs conflict") {
        CHECK_THROWS(ManyToOne<int, std::string>{
            {{1, 10, 100}, "one"},
            {{2, 20, 10}, "two"},
        });
      }
    }
  }

  TEST_CASE("adl_serializer<ManyToOne<L, R>>") {
    ManyToOne<int, std::string> deserialized = ManyToOne<int, std::string>{
        {{2, 20}, {"two"}},
        {{3}, "three"},
        {{4, 40, 400}, "four"},
    };

    nlohmann::json serialized = std::set<std::pair<int, std::string>>{
        {2, "two"},
        {3, "three"},
        {4, "four"},
        {20, "two"},
        {40, "four"},
        {400, "four"},
    };

    SUBCASE("to_json") {
      nlohmann::json result = deserialized;
      nlohmann::json correct = serialized;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      ManyToOne<int, std::string> result = serialized;
      ManyToOne<int, std::string> correct = deserialized;

      CHECK(result == correct);
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

  TEST_CASE("many_to_one_from_unstructured_relation") {
    SUBCASE("relation is many-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "odd"},
          {2, "even"},
          {3, "odd"},
      };

      ManyToOne<int, std::string> result =
          many_to_one_from_unstructured_relation(input);
      ManyToOne<int, std::string> correct = {
          {{1, 3}, "odd"},
          {{2}, "even"},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is one-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };

      ManyToOne<int, std::string> result =
          many_to_one_from_unstructured_relation(input);
      ManyToOne<int, std::string> correct = {
          {{1}, "one"},
          {{2}, "two"},
          {{3}, "three"},
      };

      CHECK(result == correct);
    }

    SUBCASE("relation is not many-to-one") {
      std::unordered_set<std::pair<int, std::string>> input = {
          {1, "one"},
          {1, "ODD"},
          {2, "two"},
          {3, "ODD"},
      };

      CHECK_THROWS(many_to_one_from_unstructured_relation(input));
    }
  }
}
