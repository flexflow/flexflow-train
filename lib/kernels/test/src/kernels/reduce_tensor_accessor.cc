#include "kernels/reduce_tensor_accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include "utils/containers/sum.h"
#include "utils/overload.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reduce_tensor_accessor_in_dims") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR accessor =
        create_3d_accessor_r_with_contents<int32_t>(
            {
                {
                    {1, 3, 2},
                    {2, 1, 5},
                },
                {
                    {4, 2, 1},
                    {8, 3, 6},
                },
            },
            cpu_allocator);

    GenericTensorAccessorW result = reduce_tensor_accessor_in_dims(
        accessor,
        {ff_dim_t{0_n}, ff_dim_t{2_n}},
        cpu_allocator,
        overload{
            [](std::vector<int> const &input_values) {
              return sum(input_values);
            },
            [](auto const &x) -> get_element_type_t<decltype(x)> { PANIC(); },
        });

    GenericTensorAccessorW correct =
        create_1d_accessor_w_with_contents<int32_t>(
            {
                1 + 3 + 2 + 4 + 2 + 1,
                2 + 1 + 5 + 8 + 3 + 6,
            },
            cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result =", format_accessor_w_contents(result)),
                  check_kv("correct=", format_accessor_w_contents(correct)));
  }

  TEST_CASE("reduce_tensor_accessor_in_all_dims") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR accessor =
        create_3d_accessor_r_with_contents<int32_t>(
            {
                {
                    {1, 3, 2},
                    {2, 1, 5},
                },
                {
                    {4, 2, 1},
                    {8, 3, 6},
                },
            },
            cpu_allocator);

    int32_t result = reduce_tensor_accessor_in_all_dims<DataType::INT32>(
        accessor,
        overload{
            [](std::vector<int> const &input_values) {
              return sum(input_values);
            },
            [](auto const &x) -> get_element_type_t<decltype(x)> { PANIC(); },
        });
    int32_t correct = 1 + 3 + 2 + 2 + 1 + 5 + 4 + 2 + 1 + 8 + 3 + 6;

    CHECK(result == correct);
  }
}
