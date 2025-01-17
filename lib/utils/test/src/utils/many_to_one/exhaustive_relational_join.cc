#include "utils/many_to_one/exhaustive_relational_join.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("exhaustive_relational_join(ManyToOne<T1, T2>, ManyToOne<T2, T3>)") {
    SUBCASE("inputs are empty") {
      ManyToOne<int, std::string> fst = {};
      ManyToOne<std::string, std::pair<int, int>> snd = {};

      ManyToOne<int, std::pair<int, int>> result = exhaustive_relational_join(fst, snd);
      ManyToOne<int, std::pair<int, int>> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("succeeds if join is exhaustive") {
      ManyToOne<int, std::string> fst = {
        {{2, 4}, "2"},
        {{3, 9, 27}, "3"},
        {{5, 25, 125}, "5"},
      };

      ManyToOne<std::string, std::pair<std::string, bool>> snd = {
        {{"2"}, {"even", true}},
        {{"3", "5"}, {"odd", false}},
      };

      ManyToOne<int, std::pair<std::string, bool>> result = exhaustive_relational_join(fst, snd);
      ManyToOne<int, std::pair<std::string, bool>> correct = {
        {{2, 4}, {"even", true}},
        {{3, 9, 27, 5, 25, 125}, {"odd", false}},
      };

      CHECK(result == correct);
    }

    SUBCASE("throws if extra R in fst") {
      ManyToOne<int, std::string> fst = {
        {{2, 4}, "2"},
        {{3, 9, 27}, "3"},
        {{5, 25, 125}, "5"},
        {{6, 36}, "6"},
      };

      ManyToOne<std::string, std::pair<std::string, bool>> snd = {
        {{"2"}, {"even", true}},
        {{"3", "5"}, {"odd", false}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("throws if extra L in snd") {
      ManyToOne<int, std::string> fst = {
        {{2, 4}, "2"},
        {{3, 9, 27}, "3"},
        {{5, 25, 125}, "5"},
      };

      ManyToOne<std::string, std::pair<std::string, bool>> snd = {
        {{"2", "6"}, {"even", true}},
        {{"3", "5"}, {"odd", false}},
      };

      CHECK_THROWS(exhaustive_relational_join(fst, snd));
    }

    SUBCASE("works even if all types are the same") {
      ManyToOne<int, int> fst = {
        {{2, 4}, 2},
        {{3, 9, 27}, 3},
        {{5, 25, 125}, 5},
      };

      ManyToOne<int, int> snd = {
        {{2}, 1},
        {{3, 5}, 0},
      };


      ManyToOne<int, int> result = exhaustive_relational_join(fst, snd);
      ManyToOne<int, int> correct = {
        {{2, 4}, 1},
        {{3, 9, 27, 5, 25, 125}, 0},
      };

      CHECK(result == correct);
    }
  }
}
