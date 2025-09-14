#include <doctest/doctest.h>
#include "op-attrs/tensor_dim_permutation.h"
#include "test/utils/rapidcheck/doctest.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("TensorDimPermutation") {
    SUBCASE("fails if constructed with a non-contiguous key set") {
      CHECK_THROWS(
        TensorDimPermutation{bidict<ff_dim_t, ff_dim_t>{
          {ff_dim_t{2_n}, ff_dim_t{0_n}},
          {ff_dim_t{0_n}, ff_dim_t{1_n}},
        }}
      );
    }

    SUBCASE("fails if constructed with a key set that doesn't start at 1") {
      CHECK_THROWS(
        TensorDimPermutation{bidict<ff_dim_t, ff_dim_t>{
          {ff_dim_t{0_n}, ff_dim_t{1_n}},
          {ff_dim_t{1_n}, ff_dim_t{2_n}},
        }}
      );
    }

    SUBCASE("can be constructed with empty bidict") {
      TensorDimPermutation p = TensorDimPermutation{bidict<ff_dim_t, ff_dim_t>{}};
      CHECK(p.num_tensor_dims() == num_tensor_dims_t{0_n});
    }

    SUBCASE("can be constructed with non-empty bidict") {
      bidict<ff_dim_t, ff_dim_t> b = bidict<ff_dim_t, ff_dim_t>{
        {ff_dim_t{0_n}, ff_dim_t{2_n}},
        {ff_dim_t{1_n}, ff_dim_t{3_n}},
        {ff_dim_t{3_n}, ff_dim_t{0_n}},
        {ff_dim_t{2_n}, ff_dim_t{1_n}},
      };

      TensorDimPermutation p = TensorDimPermutation{b};

      SUBCASE("at_l") {
        SUBCASE("key is present") {
          ff_dim_t result = p.at_l(ff_dim_t{1_n});
          ff_dim_t correct = ff_dim_t{3_n};

          CHECK(result == correct);
        }

        SUBCASE("key is not present") {
          CHECK_THROWS(p.at_l(ff_dim_t{4_n}));
        }
      }

      SUBCASE("at_r") {
        SUBCASE("key is present") {
          ff_dim_t result = p.at_r(ff_dim_t{1_n}); 
          ff_dim_t correct = ff_dim_t{2_n};

          CHECK(result == correct);
        }

        SUBCASE("key is not present") {
          CHECK_THROWS(p.at_r(ff_dim_t{4_n}));
        }
      }

      SUBCASE("num_tensor_dims") {
        num_tensor_dims_t result = p.num_tensor_dims();
        num_tensor_dims_t correct = num_tensor_dims_t{4_n};

        CHECK(result == correct);
      }

      SUBCASE("as_bidict") {
        bidict<ff_dim_t, ff_dim_t> result = p.as_bidict();     
        bidict<ff_dim_t, ff_dim_t> correct = b;

        CHECK(result == correct);
      }
    }
  }

  TEST_CASE("Arbitrary<TensorDimPermutation>") {
    RC_SUBCASE([](TensorDimPermutation) {}); 
  }
}
