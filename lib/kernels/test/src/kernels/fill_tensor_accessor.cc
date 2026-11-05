#include "kernels/fill_tensor_accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/datatype_value.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("fill_accessor_with_constant") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW result = create_random_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 4_p, 2_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    SUBCASE("filled with correct datatype") {
      fill_accessor_with_constant(result, make_float_data_type_value(12.0f));

      GenericTensorAccessorW correct =
          create_3d_accessor_w_with_contents<float>(
              {
                  {
                      {12.0, 12.0},
                      {12.0, 12.0},
                      {12.0, 12.0},
                      {12.0, 12.0},
                  },
                  {
                      {12.0, 12.0},
                      {12.0, 12.0},
                      {12.0, 12.0},
                      {12.0, 12.0},
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("filled with double but tensor type is float") {
      CHECK_THROWS(fill_accessor_with_constant(
          result, make_double_data_type_value(12.0)));
    }

    SUBCASE("filled with float but tensor type is double") {
      GenericTensorAccessorW double_result = create_random_filled_accessor_w(
          TensorShape{
              TensorDims{FFOrdered{2_p, 4_p, 2_p}},
              DataType::DOUBLE,
          },
          cpu_allocator);

      CHECK_THROWS(fill_accessor_with_constant(
          double_result, make_float_data_type_value(12.0)));
    }
  }
}
