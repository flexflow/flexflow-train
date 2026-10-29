#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "test/utils/doctest/fmt/set.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/sorted_by.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("shard_dim_idx_for_relative") {
    num_ptensor_shard_dims_t num_dims = num_ptensor_shard_dims_t{5_n};

    SUBCASE("relative > 0") {
      int relative = 1;

      parallel_tensor_dim_idx_t result =
          shard_dim_idx_for_relative(relative, num_dims);
      parallel_tensor_dim_idx_t correct = shard_dim_idx(ff_dim_t{1_n});

      CHECK(result == correct);
    }

    SUBCASE("relative < 0") {
      int relative = -2;

      parallel_tensor_dim_idx_t result =
          shard_dim_idx_for_relative(relative, num_dims);
      parallel_tensor_dim_idx_t correct = shard_dim_idx(ff_dim_t{3_n});

      CHECK(result == correct);
    }

    SUBCASE("relative == 0") {
      int relative = 0;

      parallel_tensor_dim_idx_t result =
          shard_dim_idx_for_relative(relative, num_dims);
      parallel_tensor_dim_idx_t correct = shard_dim_idx(ff_dim_t{0_n});

      CHECK(result == correct);
    }

    SUBCASE("max relative in bounds") {
      int relative = 4;

      parallel_tensor_dim_idx_t result =
          shard_dim_idx_for_relative(relative, num_dims);
      parallel_tensor_dim_idx_t correct = shard_dim_idx(ff_dim_t{4_n});

      CHECK(result == correct);
    }

    SUBCASE("min relative in bounds") {
      int relative = -5;

      parallel_tensor_dim_idx_t result =
          shard_dim_idx_for_relative(relative, num_dims);
      parallel_tensor_dim_idx_t correct = shard_dim_idx(ff_dim_t{0_n});

      CHECK(result == correct);
    }

    SUBCASE("relative out of bounds (too high)") {
      int relative = 5;

      CHECK_THROWS(shard_dim_idx_for_relative(relative, num_dims));
    }

    SUBCASE("relative out of bounds (too low)") {
      int relative = -6;

      CHECK_THROWS(shard_dim_idx_for_relative(relative, num_dims));
    }
  }

  TEST_CASE("get_parallel_tensor_dim_ordering") {
    DimOrdering<parallel_tensor_dim_idx_t> ordering =
        get_parallel_tensor_dim_ordering();

    RC_SUBCASE("is antireflexive", [&](parallel_tensor_dim_idx_t const &idx) {
      RC_ASSERT(!(idx < idx));
    });

    RC_SUBCASE("is antisymmetric",
               [&](parallel_tensor_dim_idx_t const &a,
                   parallel_tensor_dim_idx_t const &b) {
                 RC_PRE(a < b);

                 RC_ASSERT(!(b < a));
               });

    RC_SUBCASE("is transitive",
               [&](parallel_tensor_dim_idx_t const &a,
                   parallel_tensor_dim_idx_t const &b,
                   parallel_tensor_dim_idx_t const &c) {
                 RC_PRE(a < b);
                 RC_PRE(b < c);

                 RC_ASSERT(a < c);
               });

    SUBCASE("sum is less than discard") {
      bool result = ordering.lt(sum_dim_idx(), discard_copy_dim_idx());
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("discard is less than shard dim") {
      bool result =
          ordering.lt(discard_copy_dim_idx(), shard_dim_idx(ff_dim_t{0_n}));
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("shard dim 0 is less than shard dim 1") {
      bool result = ordering.lt(shard_dim_idx(ff_dim_t{0_n}),
                                shard_dim_idx(ff_dim_t{1_n}));
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("properly sorts a set of dimensions") {
      std::set<parallel_tensor_dim_idx_t> input = {
          sum_dim_idx(),
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{0_n}),
          discard_copy_dim_idx(),
      };

      std::vector<parallel_tensor_dim_idx_t> result =
          sorted_by(input, get_parallel_tensor_dim_ordering().lt);

      std::vector<parallel_tensor_dim_idx_t> correct = {
          sum_dim_idx(),
          discard_copy_dim_idx(),
          shard_dim_idx(ff_dim_t{0_n}),
          shard_dim_idx(ff_dim_t{1_n}),
      };

      CHECK(result == correct);
    }
  }

  TEST_CASE("shard_dim_idxs_for_inclusive_interval") {
    num_ptensor_shard_dims_t num_dims = num_ptensor_shard_dims_t{5_n};

    SUBCASE("start >= 0, stop >= 0, start < stop") {
      int start = 1;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop >= 0, start == stop") {
      int start = 4;
      int stop = 4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{4_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop >= 0, start >= stop") {
      int start = 3;
      int stop = 1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop >= 0, resolved(start) >= stop") {
      int start = -3;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop >= 0, resolved(start) == stop") {
      int start = -2;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop >= 0, resolved(start) >= stop") {
      int start = -1;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop < 0, start >= resolved(stop)") {
      int start = 2;
      int stop = -2;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop < 0, start == resolved(stop)") {
      int start = 4;
      int stop = -1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{4_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop < 0, start >= resolved(stop)") {
      int start = 3;
      int stop = -3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop < 0, resolved(start) >= resolved(stop)") {
      int start = -3;
      int stop = -1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
          shard_dim_idx(ff_dim_t{4_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop < 0, resolved(start) == resolved(stop)") {
      int start = -4;
      int stop = -4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{1_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop < 0, resolved(start) >= resolved(stop)") {
      int start = -1;
      int stop = -4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("max start value in bounds") {
      int start = 4;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start out of bounds (too high)") {
      int start = 5;
      int stop = 3;

      CHECK_THROWS(
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims));
    }

    SUBCASE("minimum start value in bounds") {
      int start = -5;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{0_n}),
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start out of bounds (too low)") {
      int start = -6;
      int stop = 3;

      CHECK_THROWS(
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims));
    }

    SUBCASE("maximum stop value in bounds") {
      int start = 1;
      int stop = 4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
          shard_dim_idx(ff_dim_t{4_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("stop out of bounds (too high)") {
      int start = 1;
      int stop = 5;

      CHECK_THROWS(
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims));
    }

    SUBCASE("minimum stop value in bounds") {
      int start = 1;
      int stop = -5;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("stop out of bounds (too low)") {
      int start = 1;
      int stop = -6;

      CHECK_THROWS(
          shard_dim_idxs_for_inclusive_interval(start, stop, num_dims));
    }
  }

  TEST_CASE("shard_dim_idxs_for_exclusive_interval") {
    num_ptensor_shard_dims_t num_dims = num_ptensor_shard_dims_t{5_n};

    SUBCASE("start >= 0, stop >= 0, start < stop") {
      int start = 1;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{2_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop >= 0, start == stop") {
      int start = 4;
      int stop = 4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop >= 0, start >= stop") {
      int start = 3;
      int stop = 1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop >= 0, resolved(start) >= stop") {
      int start = -3;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{2_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop >= 0, resolved(start) == stop") {
      int start = -2;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop >= 0, resolved(start) >= stop") {
      int start = -1;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop < 0, start >= resolved(stop)") {
      int start = 2;
      int stop = -1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop < 0, start == resolved(stop)") {
      int start = 4;
      int stop = -1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start >= 0, stop < 0, start >= resolved(stop)") {
      int start = 3;
      int stop = -3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop < 0, resolved(start) >= resolved(stop)") {
      int start = -3;
      int stop = -1;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop < 0, resolved(start) == resolved(stop)") {
      int start = -4;
      int stop = -4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start < 0, stop < 0, resolved(start) >= resolved(stop)") {
      int start = -1;
      int stop = -4;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("max start value in bounds") {
      int start = 4;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("start out of bounds (too high)") {
      int start = 5;
      int stop = 3;

      CHECK_THROWS(
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims));
    }

    SUBCASE("minimum start value in bounds") {
      int start = -5;
      int stop = 3;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{0_n}),
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{2_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("start out of bounds (too low)") {
      int start = -6;
      int stop = 3;

      CHECK_THROWS(
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims));
    }

    SUBCASE("maximum stop value in bounds") {
      int start = 1;
      int stop = 5;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {
          shard_dim_idx(ff_dim_t{1_n}),
          shard_dim_idx(ff_dim_t{2_n}),
          shard_dim_idx(ff_dim_t{3_n}),
          shard_dim_idx(ff_dim_t{4_n}),
      };

      CHECK(result == correct);
    }

    SUBCASE("stop out of bounds (too high)") {
      int start = 1;
      int stop = 6;

      CHECK_THROWS(
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims));
    }

    SUBCASE("minimum stop value in bounds") {
      int start = 1;
      int stop = -5;

      std::set<parallel_tensor_dim_idx_t> result =
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims);
      std::set<parallel_tensor_dim_idx_t> correct = {};

      CHECK(result == correct);
    }

    SUBCASE("stop out of bounds (too low)") {
      int start = 1;
      int stop = -6;

      CHECK_THROWS(
          shard_dim_idxs_for_exclusive_interval(start, stop, num_dims));
    }
  }
}
