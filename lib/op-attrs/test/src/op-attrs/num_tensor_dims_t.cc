#include "op-attrs/num_tensor_dims_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("ff_dim_is_valid_for_num_tensor_dims") {
    num_tensor_dims_t num_dims = num_tensor_dims_t{4_n};
    SUBCASE("is largest dim in bounds") {
      ff_dim_t dim = ff_dim_t{3_n};

      bool result = ff_dim_is_valid_for_num_tensor_dims(dim, num_dims);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("is smallest dim out of bounds") {
      ff_dim_t dim = ff_dim_t{4_n};

      bool result = ff_dim_is_valid_for_num_tensor_dims(dim, num_dims);
      bool correct = false;

      CHECK(result == correct);
    }
  }
}
