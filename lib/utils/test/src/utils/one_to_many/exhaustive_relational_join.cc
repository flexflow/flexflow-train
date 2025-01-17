#include "utils/one_to_many/exhaustive_relational_join.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("exhaustive_relational_join(OneToMany<T1, T2>, OneToMany<T2, T3>)") {
    SUBCASE("inputs are empty") {
      OneToMany<int, std::string> fst = {};
      OneToMany<std::string, std::pair<int, int>> snd = {};

      OneToMany<int, std::pair<int, int>> result = exhaustive_relational_join(fst, snd);
      OneToMany<int, std::pair<int, int>> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("succeeds if join is exhaustive") {
      OneToMany<int, std::string> fst = {
        {1, {"one", "ONE"}},
        {2, {"two"}},
        {3, {"three"}},
      };
      OneToMany<std::string, std::pair<std::string, int>> snd = {
        {"one", {{"one", 0}}},
        {"ONE", {{"ONE", 0}, {"ONE", 1}}},
        {"two", {{"two", 0}, {"two", 1}}},
        {"three", {{"three", 2}}},
      };

      OneToMany<int, std::pair<std::string, int>> result = exhaustive_relational_join(fst, snd);
      OneToMany<int, std::pair<std::string, int>> correct = {
        {1, {{"one", 0}, {"ONE", 0}, {"ONE", 1}}},
        {2, {{"two", 0}, {"two", 1}}},
        {3, {{"three", 2}}},
      };

      CHECK(result == correct);
    }

    SUBCASE("throws if extra R in fst") {
      OneToMany<int, std::string> fst = {
        {1, {"one", "One", "ONE"}},
        {2, {"two"}},
        {3, {"three"}},
      };
      OneToMany<std::string, std::pair<std::string, int>> snd = {
        {"one", {{"one", 0}}},
        {"ONE", {{"ONE", 0}, {"ONE", 1}}},
        {"two", {{"two", 0}, {"two", 1}, {"two", 2}}},
        {"three", {{"three", 2}}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("throws if extra L in snd") {
      OneToMany<int, std::string> fst = {
        {1, {"one"}},
        {2, {"two"}},
        {3, {"three"}},
      };
      OneToMany<std::string, std::pair<std::string, int>> snd = {
        {"one", {{"one", 0}}},
        {"ONE", {{"ONE", 0}, {"ONE", 1}}},
        {"two", {{"two", 0}, {"two", 1}, {"two", 2}}},
        {"three", {{"three", 2}}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("works even if all types are the same") {
      OneToMany<int, int> fst = {
        {4, {2}},
        {18*20, {18, 20}},
      };
      OneToMany<int, int> snd = {
        {2, {2}},
        {18, {3, 6}},
        {20, {4, 5}},
      };

      OneToMany<int, int> result = exhaustive_relational_join(fst, snd);
      OneToMany<int, int> correct = {
        {4, {2}},
        {18*20, {3, 4, 5, 6}},
      };

      CHECK(result == correct);
    }
  }
}
