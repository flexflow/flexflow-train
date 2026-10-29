#include "op-attrs/ff_dim_t.h"
#include "test/utils/doctest/fmt/vector.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("relative_ff_dim_t_from_ff_dim_t") {
    SUBCASE("absolute index is zero") {
      ff_dim_t ff_dim = ff_dim_t{nonnegative_int{0}};
      relative_ff_dim_t relative_ff_dim =
          relative_ff_dim_t_from_ff_dim_t(ff_dim);
      CHECK(relative_ff_dim == relative_ff_dim_t{0});
    }

    SUBCASE("absolute index is positive") {
      ff_dim_t ff_dim = ff_dim_t{nonnegative_int{1}};
      relative_ff_dim_t relative_ff_dim =
          relative_ff_dim_t_from_ff_dim_t(ff_dim);
      CHECK(relative_ff_dim == relative_ff_dim_t{1});
    }
  }

  TEST_CASE("ff_dim_range2_inclusive") {
    SUBCASE("start < end") {
      ff_dim_t start = ff_dim_t{2_n};
      ff_dim_t end = ff_dim_t{4_n};

      std::vector<ff_dim_t> result = ff_dim_range2_inclusive(start, end);
      std::vector<ff_dim_t> correct = {
          ff_dim_t{2_n},
          ff_dim_t{3_n},
          ff_dim_t{4_n},
      };

      CHECK(result == correct);
    }

    SUBCASE("start == end") {
      ff_dim_t start = ff_dim_t{4_n};
      ff_dim_t end = ff_dim_t{4_n};

      std::vector<ff_dim_t> result = ff_dim_range2_inclusive(start, end);
      std::vector<ff_dim_t> correct = {
          ff_dim_t{4_n},
      };

      CHECK(result == correct);
    }

    SUBCASE("start > end") {
      ff_dim_t start = ff_dim_t{6_n};
      ff_dim_t end = ff_dim_t{4_n};

      std::vector<ff_dim_t> result = ff_dim_range2_inclusive(start, end);
      std::vector<ff_dim_t> correct = {};

      CHECK(result == correct);
    }
  }

  TEST_CASE("ff_dim_range2_exclusive") {
    SUBCASE("start < end") {
      ff_dim_t start = ff_dim_t{2_n};
      ff_dim_t end = ff_dim_t{4_n};

      std::vector<ff_dim_t> result = ff_dim_range2_exclusive(start, end);
      std::vector<ff_dim_t> correct = {
          ff_dim_t{2_n},
          ff_dim_t{3_n},
      };

      CHECK(result == correct);
    }

    SUBCASE("start == end") {
      ff_dim_t start = ff_dim_t{4_n};
      ff_dim_t end = ff_dim_t{4_n};

      std::vector<ff_dim_t> result = ff_dim_range2_exclusive(start, end);
      std::vector<ff_dim_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start > end") {
      ff_dim_t start = ff_dim_t{7_n};
      ff_dim_t end = ff_dim_t{4_n};

      std::vector<ff_dim_t> result = ff_dim_range2_exclusive(start, end);
      std::vector<ff_dim_t> correct = {};

      CHECK(result == correct);
    }
  }
}
