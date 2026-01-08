#include "op-attrs/operator_task_space_dim_idx_t.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("operator_task_space_dim_idx_range(nonnegative_int)") {
    SUBCASE("end is zero") {
      std::set<operator_task_space_dim_idx_t> result =
          operator_task_space_dim_idx_range(nonnegative_int{0});
      std::set<operator_task_space_dim_idx_t> correct = {
          operator_task_space_dim_idx_t{nonnegative_int{0}}};
    }
  }
}
