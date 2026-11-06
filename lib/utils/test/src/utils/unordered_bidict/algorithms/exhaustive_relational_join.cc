#include "utils/unordered_bidict/algorithms/exhaustive_relational_join.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("exhaustive_relational_join(unordered_bidict<T1, T2>, "
            "unordered_bidict<T2, T3>)") {
    SUBCASE("inputs are empty") {
      unordered_bidict<int, std::string> fst = {};
      unordered_bidict<std::string, std::pair<int, int>> snd = {};

      unordered_bidict<int, std::pair<int, int>> result =
          exhaustive_relational_join(fst, snd);
      unordered_bidict<int, std::pair<int, int>> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("join is exhaustive") {
      unordered_bidict<int, std::string> fst = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };
      unordered_bidict<std::string, std::pair<int, int>> snd = {
          {"one", {2, 0}}, {"two", {3, 1}}, {"three", {4, 2}}};

      unordered_bidict<int, std::pair<int, int>> result =
          exhaustive_relational_join(fst, snd);
      unordered_bidict<int, std::pair<int, int>> correct = {
          {1, {2, 0}},
          {2, {3, 1}},
          {3, {4, 2}},
      };

      CHECK(result == correct);
    }

    SUBCASE("extra relation in fst") {
      unordered_bidict<int, std::string> fst = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };
      unordered_bidict<std::string, std::pair<int, int>> snd = {
          {"one", {2, 0}},
          {"two", {3, 1}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("extra relation in snd") {
      unordered_bidict<int, std::string> fst = {
          {1, "one"},
          {3, "three"},
      };
      unordered_bidict<std::string, std::pair<int, int>> snd = {
          {"one", {2, 0}},
          {"two", {3, 1}},
          {"three", {4, 2}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("same number of relations in fst and snd, but not matching") {
      unordered_bidict<int, std::string> fst = {
          {1, "one"},
          {2, "two"},
          {3, "three"},
      };
      unordered_bidict<std::string, std::pair<int, int>> snd = {
          {"one", {2, 0}},
          {"three", {4, 2}},
          {"four", {5, 3}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("works even if all the types are the same") {
      unordered_bidict<int, int> fst = {
          {1, 2},
          {2, 3},
      };
      unordered_bidict<int, int> snd = {
          {2, 3},
          {3, 4},
      };

      unordered_bidict<int, int> result = exhaustive_relational_join(fst, snd);
      unordered_bidict<int, int> correct = {
          {1, 3},
          {2, 4},
      };

      CHECK(result == correct);
    }
  }
}
