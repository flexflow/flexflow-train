#include "kernels/reduce_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("reduce_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {3, 2, 4},
                    {6, 0, 1},
                },
                {
                    {-1, 2, -3},
                    {2, 5, 4},
                },
            },
            {
                {
                    {8, 3, 6},
                    {1, -1, -2},
                },
                {
                    {1, 1, 5},
                    {2, 3, -5},
                },
            },
        },
        cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 2_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    SUBCASE("op = SUM") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::SUM,
          /*keepdims=*/false,
      };

      reduce_cpu_forward_kernel(attrs, input, result);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {
                      3 + 2 + 4 + (-1) + 2 + (-3),
                      6 + 0 + 1 + 2 + 5 + 4,
                  },
                  {
                      8 + 3 + 6 + 1 + 1 + 5,
                      1 + (-1) + (-2) + 2 + 3 + (-5),
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("op = AVG") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::AVG,
          /*keepdims=*/false,
      };

      reduce_cpu_forward_kernel(attrs, input, result);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {
                      (3 + 2 + 4 + (-1) + 2 + (-3)) / 6.0f,
                      (6 + 0 + 1 + 2 + 5 + 4) / 6.0f,
                  },
                  {
                      (8 + 3 + 6 + 1 + 1 + 5) / 6.0f,
                      (1 + (-1) + (-2) + 2 + 3 + (-5)) / 6.0f,
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }

  TEST_CASE("reduce_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR output_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {3, 2},
                {-2, 0},
            },
            cpu_allocator);

    GenericTensorAccessorW result = create_zero_filled_accessor_w(
        TensorShape{
            TensorDims{FFOrdered{2_p, 2_p, 2_p, 3_p}},
            DataType::FLOAT,
        },
        cpu_allocator);

    SUBCASE("op = SUM") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::SUM,
          /*keepdims=*/false,
      };

      reduce_cpu_backward_kernel(attrs, output_grad, result);

      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {
                          {1, 1, 1},
                          {1, 1, 1},
                      },
                      {
                          {1, 1, 1},
                          {1, 1, 1},
                      },
                  },
                  {
                      {
                          {1, 1, 1},
                          {1, 1, 1},
                      },
                      {
                          {1, 1, 1},
                          {1, 1, 1},
                      },
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("op = AVG") {
      ReduceAttrs attrs = ReduceAttrs{
          /*axes=*/nonempty_set<ff_dim_t>{
              ff_dim_t{1_n},
              ff_dim_t{3_n},
          },
          /*op=*/ReduceOp::AVG,
          /*keepdims=*/false,
      };

      reduce_cpu_backward_kernel(attrs, output_grad, result);

      float x = 1.0 / 6.0;
      GenericTensorAccessorR correct =
          create_4d_accessor_r_with_contents<float>(
              {
                  {
                      {
                          {x, x, x},
                          {x, x, x},
                      },
                      {
                          {x, x, x},
                          {x, x, x},
                      },
                  },
                  {
                      {
                          {x, x, x},
                          {x, x, x},
                      },
                      {
                          {x, x, x},
                          {x, x, x},
                      },
                  },
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }
}
