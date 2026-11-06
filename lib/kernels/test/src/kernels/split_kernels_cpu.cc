#include "kernels/split_kernels_cpu.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/create_zero_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/split_kernels.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("split_cpu_forward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 3, 2, 12, -1, 2},
            {2, 1, 5, 9, 5, 10},
        },
        cpu_allocator);

    SplitAttrs attrs = SplitAttrs{
        /*splits=*/{3_p, 1_p, 2_p},
        /*axis=*/ff_dim_t{1_n},
    };

    auto mk_output_shape = [&](positive_int axis_size) -> TensorShape {
      return TensorShape{
          TensorDims{
              FFOrdered<positive_int>{
                  2_p,
                  axis_size,
              },
          },
          DataType::FLOAT,
      };
    };

    GenericTensorAccessorW output_0 =
        create_zero_filled_accessor_w(mk_output_shape(3_p), cpu_allocator);

    GenericTensorAccessorW output_1 =
        create_zero_filled_accessor_w(mk_output_shape(1_p), cpu_allocator);

    GenericTensorAccessorW output_2 =
        create_zero_filled_accessor_w(mk_output_shape(2_p), cpu_allocator);

    split_cpu_forward_kernel(
        /*attrs=*/attrs,
        /*input=*/input,
        /*outputs=*/{output_0, output_1, output_2});

    GenericTensorAccessorR correct_0 =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2},
                {2, 1, 5},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_1 =
        create_2d_accessor_r_with_contents<float>(
            {
                {12},
                {9},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_2 =
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, 2},
                {5, 10},
            },
            cpu_allocator);

    CHECK_MESSAGE(accessors_are_equal(output_0, correct_0),
                  check_kv("output_0", format_accessor_w_contents(output_0)));
    CHECK_MESSAGE(accessors_are_equal(output_1, correct_1),
                  check_kv("output_1", format_accessor_w_contents(output_1)));
    CHECK_MESSAGE(accessors_are_equal(output_2, correct_2),
                  check_kv("output_2", format_accessor_w_contents(output_2)));
  }

  TEST_CASE("split_cpu_backward_kernel") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SplitAttrs attrs = SplitAttrs{
        /*splits=*/{3_p, 1_p, 2_p},
        /*axis=*/ff_dim_t{1_n},
    };

    GenericTensorAccessorR output_grad_0 =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2},
                {2, 1, 5},
            },
            cpu_allocator);

    GenericTensorAccessorR output_grad_1 =
        create_2d_accessor_r_with_contents<float>(
            {
                {12},
                {9},
            },
            cpu_allocator);

    GenericTensorAccessorR output_grad_2 =
        create_2d_accessor_r_with_contents<float>(
            {
                {-1, 2},
                {5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorR correct_input_grad =
        create_2d_accessor_r_with_contents<float>(
            {
                {1, 3, 2, 12, -1, 2},
                {2, 1, 5, 9, 5, 10},
            },
            cpu_allocator);

    GenericTensorAccessorW input_grad =
        create_zero_filled_accessor_w(correct_input_grad.shape, cpu_allocator);

    split_cpu_backward_kernel(
        /*attrs=*/attrs,
        /*output_grads=*/{output_grad_0, output_grad_1, output_grad_2},
        /*input_grad=*/input_grad);

    CHECK_MESSAGE(
        accessors_are_equal(input_grad, correct_input_grad),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));
  }
}

static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_2d_accessor_r_with_contents<float>(
      {
          {1, 2, 3, 4, 5, 6},
          {-1, -2, -3, -4, -5, -6},
          {0.5, 0.25, 0.125, 10, 20, 30},
          {7, 8, 9, 0.75, 0.5, 0.25},
      },
      allocator);
}

// Initialize with nonzero values to verify replacement.
static GenericTensorAccessorW make_input_grad(Allocator &allocator) {
  return create_2d_accessor_w_with_contents<float>(
      {
          {-3, -1, 1, 3, 5, 7},
          {9, 11, 13, 15, 17, 19},
          {21, 23, 25, 27, 29, 31},
          {33, 35, 37, 39, 41, 43},
      },
      allocator);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("split_cpu_forward_kernel (additional shapes)") {
    Allocator allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);

    SUBCASE("axis = 0") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{1_p, 3_p},
          /*axis=*/ff_dim_t{0_n},
      };

      // Intentionally randomize these tensors so we can be confident we never
      // read them
      std::vector<GenericTensorAccessorW> outputs = {
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{1_p, 6_p}}, DataType::FLOAT},
              allocator),
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{3_p, 6_p}}, DataType::FLOAT},
              allocator),
      };

      split_forward_kernel(device_stream_t{std::monostate{}},
                           /*attrs=*/attrs,
                           /*input=*/input,
                           /*outputs=*/outputs);

      std::vector<GenericTensorAccessorR> correct = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3, 4, 5, 6},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {-1, -2, -3, -4, -5, -6},
                  {0.5, 0.25, 0.125, 10, 20, 30},
                  {7, 8, 9, 0.75, 0.5, 0.25},
              },
              allocator),
      };

      for (int i = 0; i < outputs.size(); i++) {
        CHECK_MESSAGE(
            accessors_are_equal(outputs.at(i), correct.at(i)),
            check_kv("output", format_accessor_w_contents(outputs.at(i))));
      }
    }

    SUBCASE("axis = 1") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{2_p, 1_p, 3_p},
          /*axis=*/ff_dim_t{1_n},
      };

      // Intentionally randomize these tensors so we can be confident we never
      // read them
      std::vector<GenericTensorAccessorW> outputs = {
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{4_p, 2_p}}, DataType::FLOAT},
              allocator),
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{4_p, 1_p}}, DataType::FLOAT},
              allocator),
          create_random_filled_accessor_w(
              TensorShape{TensorDims{FFOrdered{4_p, 3_p}}, DataType::FLOAT},
              allocator),
      };

      split_forward_kernel(device_stream_t{std::monostate{}},
                           /*attrs=*/attrs,
                           /*input=*/input,
                           /*outputs=*/outputs);

      std::vector<GenericTensorAccessorR> correct = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2},
                  {-1, -2},
                  {0.5, 0.25},
                  {7, 8},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {3},
                  {-3},
                  {0.125},
                  {9},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {4, 5, 6},
                  {-4, -5, -6},
                  {10, 20, 30},
                  {0.75, 0.5, 0.25},
              },
              allocator),
      };

      for (int i = 0; i < outputs.size(); i++) {
        CHECK_MESSAGE(
            accessors_are_equal(outputs.at(i), correct.at(i)),
            check_kv("output", format_accessor_w_contents(outputs.at(i))));
      }
    }
  }

  TEST_CASE("split_cpu_backward_kernel (replacement, additional shapes)") {
    Allocator allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);

    SUBCASE("axis = 0") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{1_p, 3_p},
          /*axis=*/ff_dim_t{0_n},
      };

      std::vector<GenericTensorAccessorR> outputs = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3, 4, 5, 6},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {-1, -2, -3, -4, -5, -6},
                  {0.5, 0.25, 0.125, 10, 20, 30},
                  {7, 8, 9, 0.75, 0.5, 0.25},
              },
              allocator),
      };

      std::vector<GenericTensorAccessorR> output_grads = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.75, -0.5, -0.25, 0, 0.25, 0.5},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {9.25, 9.5, 9.75, 10, 10.25, 10.5},
                  {10.75, 11, 11.25, 11.5, 11.75, 12},
                  {12.25, 12.5, 12.75, 13, 13.25, 13.5},
              },
              allocator),
      };

      GenericTensorAccessorW input_grad = make_input_grad(allocator);

      split_backward_kernel(device_stream_t{std::monostate{}},
                            /*attrs=*/attrs,
                            /*output_grads=*/output_grads,
                            /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.75, -0.5, -0.25, 0, 0.25, 0.5},
                  {9.25, 9.5, 9.75, 10, 10.25, 10.5},
                  {10.75, 11, 11.25, 11.5, 11.75, 12},
                  {12.25, 12.5, 12.75, 13, 13.25, 13.5},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }

    SUBCASE("axis = 1") {
      SplitAttrs attrs = SplitAttrs{
          /*splits=*/{2_p, 1_p, 3_p},
          /*axis=*/ff_dim_t{1_n},
      };

      std::vector<GenericTensorAccessorR> outputs = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2},
                  {-1, -2},
                  {0.5, 0.25},
                  {7, 8},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {3},
                  {-3},
                  {0.125},
                  {9},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {4, 5, 6},
                  {-4, -5, -6},
                  {10, 20, 30},
                  {0.75, 0.5, 0.25},
              },
              allocator),
      };

      std::vector<GenericTensorAccessorR> output_grads = {
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.75, -0.5},
                  {-0.25, 0},
                  {0.25, 0.5},
                  {0.75, 1},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {9.25},
                  {9.5},
                  {9.75},
                  {10},
              },
              allocator),
          create_2d_accessor_r_with_contents<float>(
              {
                  {19.25, 19.5, 19.75},
                  {20, 20.25, 20.5},
                  {20.75, 21, 21.25},
                  {21.5, 21.75, 22},
              },
              allocator),
      };

      GenericTensorAccessorW input_grad = make_input_grad(allocator);

      split_backward_kernel(device_stream_t{std::monostate{}},
                            /*attrs=*/attrs,
                            /*output_grads=*/output_grads,
                            /*input_grad=*/input_grad);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {-0.75, -0.5, 9.25, 19.25, 19.5, 19.75},
                  {-0.25, 0, 9.5, 20, 20.25, 20.5},
                  {0.25, 0.5, 9.75, 20.75, 21, 21.25},
                  {0.75, 1, 10, 21.5, 21.75, 22},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }
}
