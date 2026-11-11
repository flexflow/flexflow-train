#include "kernels/transpose_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/transpose_kernels.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transpose_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = TransposeAttrs{
        /*permutation=*/TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                {ff_dim_t{0_n}, ff_dim_t{3_n}},
                {ff_dim_t{1_n}, ff_dim_t{0_n}},
                {ff_dim_t{2_n}, ff_dim_t{2_n}},
                {ff_dim_t{3_n}, ff_dim_t{1_n}},
            },
        },
    };

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {3, 3, 6},
                    {0, -1, 0.75},
                },
                {
                    {1, 2, -5},
                    {12, 0, 12},
                },
                {
                    {8, 0, -0.5},
                    {1, 9, 2},
                },
                {
                    {1, 5, 1},
                    {2, 9, 0},
                },
            },
        },
        cpu_allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                3_p,
                1_p,
                2_p,
                4_p,
            },
        },
        DataType::FLOAT,
    };

    GenericTensorAccessorW result =
        create_random_filled_accessor_w(output_shape, cpu_allocator);

    transpose_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/result);

    GenericTensorAccessorR correct = create_4d_accessor_w_with_contents<float>(
        {
            {
                {
                    {3, 1, 8, 1},
                    {0, 12, 1, 2},
                },
            },
            {
                {
                    {3, 2, 0, 5},
                    {-1, 0, 9, 9},
                },
            },
            {
                {
                    {6, -5, -0.5, 1},
                    {0.75, 12, 2, 0},
                },
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }

  TEST_CASE("transpose_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = TransposeAttrs{
        /*permutation=*/TensorDimPermutation{
            bidict<ff_dim_t, ff_dim_t>{
                {ff_dim_t{0_n}, ff_dim_t{3_n}},
                {ff_dim_t{1_n}, ff_dim_t{0_n}},
                {ff_dim_t{2_n}, ff_dim_t{2_n}},
                {ff_dim_t{3_n}, ff_dim_t{1_n}},
            },
        },
    };

    GenericTensorAccessorR out_grad = create_4d_accessor_w_with_contents<float>(
        {
            {
                {
                    {1, 2, 9, -1},
                    {2, 3, 2, 4},
                },
            },
            {
                {
                    {0.1, 6, 2, 1},
                    {-1, 0, 0, 1},
                },
            },
            {
                {
                    {2, -5, -0.5, 1},
                    {0, 12, 0, 12},
                },
            },
        },
        cpu_allocator);

    TensorShape input_grad_shape = TensorShape{
        TensorDims{
            FFOrdered<positive_int>{
                1_p,
                4_p,
                2_p,
                3_p,
            },
        },
        DataType::FLOAT,
    };

    GenericTensorAccessorW result =
        create_random_filled_accessor_w(input_grad_shape, cpu_allocator);

    transpose_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*out_grad=*/out_grad,
        /*in_grad=*/result);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 0.1, 2},
                    {2, -1, 0},
                },
                {
                    {2, 6, -5},
                    {3, 0, 12},
                },
                {
                    {9, 2, -0.5},
                    {2, 0, 0},
                },
                {
                    {-1, 1, 1},
                    {4, 1, 12},
                },
            },
        },
        cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(result, correct),
                  check_kv("result", format_accessor_w_contents(result)));
  }
}

// The permutation below maps each output dim to the input dim it comes from
// (i.e. `output[d] = input[permutation.at_l(d)]`), so it is equivalent to
// torch.permute(input, (1, 2, 0)).
static TransposeAttrs make_test_attrs() {
  return TransposeAttrs{
      TensorDimPermutation{
          bidict<ff_dim_t, ff_dim_t>{
              {ff_dim_t{0_n}, ff_dim_t{1_n}},
              {ff_dim_t{1_n}, ff_dim_t{2_n}},
              {ff_dim_t{2_n}, ff_dim_t{0_n}},
          },
      },
  };
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("transpose_cpu_forward_kernel (additional shapes)") {
    Allocator allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = make_test_attrs();

    GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
        {
            {
                {0.25, 0.5, 0.75, 1.0},
                {1.25, 1.5, 1.75, 2.0},
                {2.25, 2.5, 2.75, 3.0},
            },
            {
                {3.25, 3.5, 3.75, 4.0},
                {4.25, 4.5, 4.75, 5.0},
                {5.25, 5.5, 5.75, 6.0},
            },
        },
        allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{3_p, 4_p, 2_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    transpose_forward_kernel(device_stream_t{std::monostate{}},
                             /*attrs=*/attrs,
                             /*input=*/input,
                             /*output=*/output);

    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<float>(
        {
            {{0.25, 3.25}, {0.5, 3.5}, {0.75, 3.75}, {1.0, 4.0}},
            {{1.25, 4.25}, {1.5, 4.5}, {1.75, 4.75}, {2.0, 5.0}},
            {{2.25, 5.25}, {2.5, 5.5}, {2.75, 5.75}, {3.0, 6.0}},
        },
        allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("transpose_cpu_backward_kernel (replacement, additional shapes)") {
    Allocator allocator = create_local_cpu_memory_allocator();

    TransposeAttrs attrs = make_test_attrs();

    GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
        {
            {
                {0.25, 0.5, 0.75, 1.0},
                {1.25, 1.5, 1.75, 2.0},
                {2.25, 2.5, 2.75, 3.0},
            },
            {
                {3.25, 3.5, 3.75, 4.0},
                {4.25, 4.5, 4.75, 5.0},
                {5.25, 5.5, 5.75, 6.0},
            },
        },
        allocator);

    GenericTensorAccessorR output = create_3d_accessor_r_with_contents<float>(
        {
            {{0.25, 3.25}, {0.5, 3.5}, {0.75, 3.75}, {1.0, 4.0}},
            {{1.25, 4.25}, {1.5, 4.5}, {1.75, 4.75}, {2.0, 5.0}},
            {{2.25, 5.25}, {2.5, 5.5}, {2.75, 5.75}, {3.0, 6.0}},
        },
        allocator);

    GenericTensorAccessorR output_grad =
        create_3d_accessor_r_with_contents<float>(
            {
                {{-0.875, -0.75},
                 {-0.625, -0.5},
                 {-0.375, -0.25},
                 {-0.125, 0.0}},
                {{0.125, 0.25}, {0.375, 0.5}, {0.625, 0.75}, {0.875, 1.0}},
                {{1.125, 1.25}, {1.375, 1.5}, {1.625, 1.75}, {1.875, 2.0}},
            },
            allocator);

    // Initialize with nonzero values to verify replacement.
    GenericTensorAccessorW input_grad =
        create_3d_accessor_w_with_contents<float>(
            {
                {
                    {-2.5, -2.0, -1.5, -1.0},
                    {-0.5, 0.0, 0.5, 1.0},
                    {1.5, 2.0, 2.5, 3.0},
                },
                {
                    {3.5, 4.0, 4.5, 5.0},
                    {5.5, 6.0, 6.5, 7.0},
                    {7.5, 8.0, 8.5, 9.0},
                },
            },
            allocator);

    transpose_backward_kernel(device_stream_t{std::monostate{}},
                              /*attrs=*/attrs,
                              /*output_grad=*/output_grad,
                              /*input_grad=*/input_grad);

    GenericTensorAccessorR correct = create_3d_accessor_r_with_contents<float>(
        {
            {
                {-0.875, -0.625, -0.375, -0.125},
                {0.125, 0.375, 0.625, 0.875},
                {1.125, 1.375, 1.625, 1.875},
            },
            {
                {-0.75, -0.5, -0.25, 0},
                {0.25, 0.5, 0.75, 1},
                {1.25, 1.5, 1.75, 2},
            },
        },
        allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}
