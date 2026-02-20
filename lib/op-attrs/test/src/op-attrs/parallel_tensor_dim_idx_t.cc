#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test/utils/rapidcheck.h"
#include "utils/containers/sorted_by.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
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
      std::unordered_set<parallel_tensor_dim_idx_t> input = {
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
}
