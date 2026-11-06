#include "utils/benchmark_utils/random_unordered_bidict.h"
#include "utils/benchmark_utils/random_hex_string.h"
#include "utils/benchmark_utils/random_int.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("random_unordered_bidict") {
    auto random_l = [](std::mt19937 &gen) -> std::string {
      return random_hex_string(gen, 10_n);
    };

    auto random_r = [](std::mt19937 &gen) -> int { return random_int(gen); };

    SUBCASE("returns the correct size") {
      std::mt19937 gen1;

      gen1.seed(5);

      unordered_bidict<std::string, int> result =
          random_unordered_bidict(gen1, 6_n, random_l, random_r);

      CHECK(result.size() == 6);
    }

    SUBCASE("is deterministic with respect to gen") {
      std::mt19937 gen1;

      gen1.seed(5);

      SUBCASE("same gens produce the same value") {
        std::mt19937 gen2;
        gen2.seed(5);

        unordered_bidict<std::string, int> result1 =
            random_unordered_bidict(gen1, 10_n, random_l, random_r);
        unordered_bidict<std::string, int> result2 =
            random_unordered_bidict(gen2, 10_n, random_l, random_r);

        CHECK(result1 == result2);
      }

      SUBCASE("different gens produce different values") {
        std::mt19937 gen2;
        gen2.seed(8);

        unordered_bidict<std::string, int> result1 =
            random_unordered_bidict(gen1, 10_n, random_l, random_r);
        unordered_bidict<std::string, int> result2 =
            random_unordered_bidict(gen2, 10_n, random_l, random_r);

        CHECK(result1 != result2);
      }

      SUBCASE("repeated use of the same gen produces different values") {
        unordered_bidict<std::string, int> result1 =
            random_unordered_bidict(gen1, 10_n, random_l, random_r);
        unordered_bidict<std::string, int> result2 =
            random_unordered_bidict(gen1, 10_n, random_l, random_r);

        CHECK(result1 != result2);
      }
    }
  }
}
