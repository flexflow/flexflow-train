#include "kernels/upsample_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("upsample_cpu_forward_kernel") {
    Allocator allocator = create_local_cpu_memory_allocator();

    UpsampleAttrs attrs = UpsampleAttrs{
        /*scale_factor=*/int_ge_two{2},
        /*mode=*/UpsampleMode::NEAREST,
    };

    // NCHW
    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 2, 3},
                    {4, 5, 6},
                },
                {
                    {-1, -2, -3},
                    {0.5, 0.25, 0.125},
                },
            },
        },
        allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 2_p, 4_p, 6_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    upsample_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 1, 2, 2, 3, 3},
                    {1, 1, 2, 2, 3, 3},
                    {4, 4, 5, 5, 6, 6},
                    {4, 4, 5, 5, 6, 6},
                },
                {
                    {-1, -1, -2, -2, -3, -3},
                    {-1, -1, -2, -2, -3, -3},
                    {0.5, 0.5, 0.25, 0.25, 0.125, 0.125},
                    {0.5, 0.5, 0.25, 0.25, 0.125, 0.125},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("upsample_cpu_backward_kernel") {
    Allocator allocator = create_local_cpu_memory_allocator();

    UpsampleAttrs attrs = UpsampleAttrs{
        /*scale_factor=*/int_ge_two{2},
        /*mode=*/UpsampleMode::NEAREST,
    };

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 2, 3},
                    {4, 5, 6},
                },
                {
                    {-1, -2, -3},
                    {0.5, 0.25, 0.125},
                },
            },
        },
        allocator);

    GenericTensorAccessorR output = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 1, 2, 2, 3, 3},
                    {1, 1, 2, 2, 3, 3},
                    {4, 4, 5, 5, 6, 6},
                    {4, 4, 5, 5, 6, 6},
                },
                {
                    {-1, -1, -2, -2, -3, -3},
                    {-1, -1, -2, -2, -3, -3},
                    {0.5, 0.5, 0.25, 0.25, 0.125, 0.125},
                    {0.5, 0.5, 0.25, 0.25, 0.125, 0.125},
                },
            },
        },
        allocator);

    GenericTensorAccessorR output_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {
                        {-0.875, -0.75, -0.625, -0.5, -0.375, -0.25},
                        {-0.125, 0, 0.125, 0.25, 0.375, 0.5},
                        {0.625, 0.75, 0.875, 1, 1.125, 1.25},
                        {1.375, 1.5, 1.625, 1.75, 1.875, 2},
                    },
                    {
                        {2.125, 2.25, 2.375, 2.5, 2.625, 2.75},
                        {2.875, 3, 3.125, 3.25, 3.375, 3.5},
                        {3.625, 3.75, 3.875, 4, 4.125, 4.25},
                        {4.375, 4.5, 4.625, 4.75, 4.875, 5},
                    },
                },
            },
            allocator);

    // Initialize with nonzero values to verify replacement.
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {
                        {-7, -4, -1},
                        {2, 5, 8},
                    },
                    {
                        {11, 14, 17},
                        {20, 23, 26},
                    },
                },
            },
            allocator);

    upsample_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {-1.75, -0.75, 0.25},
                    {4.25, 5.25, 6.25},
                },
                {
                    {10.25, 11.25, 12.25},
                    {16.25, 17.25, 18.25},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("upsample_cpu_forward_kernel (scale_factor = 3)") {

    Allocator allocator = create_local_cpu_memory_allocator();

    UpsampleAttrs attrs{int_ge_two{3}, UpsampleMode::NEAREST};

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 2},
                    {-3, 0.5},
                },
            },
        },
        allocator);

    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{1_p, 1_p, 6_p, 6_p}},
        DataType::FLOAT,
    };

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(output_shape, allocator);

    upsample_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*output=*/output);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));
  }

  TEST_CASE("upsample_cpu_backward_kernel (scale_factor = 3)") {

    Allocator allocator = create_local_cpu_memory_allocator();

    UpsampleAttrs attrs{int_ge_two{3}, UpsampleMode::NEAREST};

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 2},
                    {-3, 0.5},
                },
            },
        },
        allocator);

    GenericTensorAccessorR output = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {1, 1, 1, 2, 2, 2},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                    {-3, -3, -3, 0.5, 0.5, 0.5},
                },
            },
        },
        allocator);

    GenericTensorAccessorR output_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {
                        {-3.75, -3.5, -3.25, -3, -2.75, -2.5},
                        {-2.25, -2, -1.75, -1.5, -1.25, -1},
                        {-0.75, -0.5, -0.25, 0, 0.25, 0.5},
                        {0.75, 1, 1.25, 1.5, 1.75, 2},
                        {2.25, 2.5, 2.75, 3, 3.25, 3.5},
                        {3.75, 4, 4.25, 4.5, 4.75, 5},
                    },
                },
            },
            allocator);

    // Initialize with nonzero values to verify replacement.
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {
                        {100, 200},
                        {300, 400},
                    },
                },
            },
            allocator);

    upsample_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grad=*/output_grad,
        /*input_grad=*/input_grad);

    GenericTensorAccessorR correct = create_4d_accessor_r_with_contents<float>(
        {
            {
                {
                    {-18, -11.25},
                    {22.5, 29.25},
                },
            },
        },
        allocator);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}
