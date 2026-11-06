#include "utils/unordered_bidict/algorithms/compose_unordered_bidicts.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compose_unordered_bidicts") {
    SUBCASE("correct usage") {
      unordered_bidict<int, std::string> fst =
          unordered_bidict<int, std::string>{
              {1, "one"},
              {2, "two"},
          };

      unordered_bidict<std::string, std::pair<std::string, std::string>> snd =
          unordered_bidict<std::string, std::pair<std::string, std::string>>{
              {"one", {"one", "ONE"}},
              {"two", {"two", "TWO"}},
          };

      unordered_bidict<int, std::pair<std::string, std::string>> result =
          compose_unordered_bidicts(fst, snd);

      unordered_bidict<int, std::pair<std::string, std::string>> correct =
          unordered_bidict<int, std::pair<std::string, std::string>>{
              {1, {"one", "ONE"}},
              {2, {"two", "TWO"}},
          };

      CHECK(result == correct);
    }

    SUBCASE("domains don't match") {
      unordered_bidict<int, std::string> fst =
          unordered_bidict<int, std::string>{
              {1, "one"},
              {2, "two"},
              {4, "four"},
          };

      unordered_bidict<std::string, std::pair<std::string, std::string>> snd =
          unordered_bidict<std::string, std::pair<std::string, std::string>>{
              {"one", {"one", "ONE"}},
              {"two", {"two", "TWO"}},
              {"three", {"three", "THREE"}},
          };

      CHECK_THROWS(compose_unordered_bidicts(fst, snd));
    }
  }
}
