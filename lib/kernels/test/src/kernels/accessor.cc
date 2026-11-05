#include "kernels/accessor.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("calculate_accessor_offset") {
    SUBCASE("one dimension") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{4_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              13_p,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 4_n;

      CHECK(result == correct);
    }

    SUBCASE("2d tensor is row-major") {
      positive_int num_rows = 5_p;
      positive_int num_cols = 6_p;

      TensorDims shape = TensorDims{
          FFOrdered{
              num_rows,
              num_cols,
          },
      };

      CHECK(calculate_accessor_offset(TensorDimsCoord{FFOrdered{0_n, 0_n}},
                                      shape) == 0_n);
      CHECK(calculate_accessor_offset(TensorDimsCoord{FFOrdered{1_n, 0_n}},
                                      shape) == num_cols);
      CHECK(calculate_accessor_offset(TensorDimsCoord{FFOrdered{0_n, 1_n}},
                                      shape) == 1_p);
    }

    SUBCASE("multiple dimensions") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{2_n, 4_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              5_p,
              6_p,
          },
      };

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 2_n * 6_n + 4_n;

      CHECK(result == correct);
    }

    SUBCASE("zero dimensions") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered<nonnegative_int>{}};
      TensorDims shape = TensorDims{FFOrdered<positive_int>{}};

      nonnegative_int result = calculate_accessor_offset(indices, shape);
      nonnegative_int correct = 0_n;

      CHECK(result == correct);
    }

    SUBCASE("index and shape dimensions do not match") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{1_n, 2_n, 4_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              5_p,
              6_p,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }

    SUBCASE("out of bounds index") {
      TensorDimsCoord indices = TensorDimsCoord{FFOrdered{2_n, 6_n}};
      TensorDims shape = TensorDims{
          FFOrdered{
              5_p,
              6_p,
          },
      };

      CHECK_THROWS(calculate_accessor_offset(indices, shape));
    }
  }

  TEST_CASE("accessor_get_only_value") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("returns the value if the accessor only contains one value") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
              },
          },
          cpu_allocator);

      float result = accessor_get_only_value<DataType::FLOAT>(input);
      float correct = 12;

      CHECK(result == correct);
    }

    SUBCASE("throws an error if the requested type does not match the type in "
            "the accessor") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
              },
          },
          cpu_allocator);

      CHECK_THROWS(accessor_get_only_value<DataType::DOUBLE>(input));
    }

    SUBCASE("throws an error if the accessor contains multiple values") {
      GenericTensorAccessorR input = create_3d_accessor_r_with_contents<float>(
          {
              {
                  {12},
                  {12},
              },
          },
          cpu_allocator);

      CHECK_THROWS(accessor_get_only_value<DataType::FLOAT>(input));
    }
  }

  TEST_CASE("copy_accessor_data_to_l_from_r") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 3, 2, 6},
            {2, 1, 5, 4},
        },
        cpu_allocator);

    auto mk_output = [&](TensorShape const &shape) -> GenericTensorAccessorW {
      return create_random_filled_accessor_w(shape, cpu_allocator);
    };

    SUBCASE("throws if the tensors are not the same size") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{2_p, 5_p},
          },
          DataType::FLOAT,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      CHECK_THROWS(copy_accessor_data_to_l_from_r(result, input));
    }

    SUBCASE("throws if the tensors do not have the same datatype") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{2_p, 4_p},
          },
          DataType::INT32,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      CHECK_THROWS(copy_accessor_data_to_l_from_r(result, input));
    }

    SUBCASE("duplicates the tensor if they are the same shape") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{2_p, 4_p},
          },
          DataType::FLOAT,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      copy_accessor_data_to_l_from_r(result, input);

      GenericTensorAccessorR correct = input;

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("allows tensors to not be the same shape in terms of dims") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{4_p, 2_p},
          },
          DataType::FLOAT,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      copy_accessor_data_to_l_from_r(result, input);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 3},
                  {2, 6},
                  {2, 1},
                  {5, 4},
              },
              cpu_allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }

  TEST_CASE("copy_accessor_data_to_l_from_r_strict") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorR input = create_2d_accessor_r_with_contents<float>(
        {
            {1, 3, 2, 6},
            {2, 1, 5, 4},
        },
        cpu_allocator);

    auto mk_output = [&](TensorShape const &shape) -> GenericTensorAccessorW {
      return create_random_filled_accessor_w(shape, cpu_allocator);
    };

    SUBCASE("throws if the tensors are not the same size") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{2_p, 5_p},
          },
          DataType::FLOAT,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      CHECK_THROWS(copy_accessor_data_to_l_from_r_strict(result, input));
    }

    SUBCASE("throws if the tensors do not have the same datatype") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{2_p, 4_p},
          },
          DataType::INT32,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      CHECK_THROWS(copy_accessor_data_to_l_from_r_strict(result, input));
    }

    SUBCASE("duplicates the tensor if they are the same shape") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{2_p, 4_p},
          },
          DataType::FLOAT,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      copy_accessor_data_to_l_from_r_strict(result, input);

      GenericTensorAccessorR correct = input;

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("does not allow tensors with equal size and datatype but different "
            "dims") {
      TensorShape output_shape = TensorShape{
          TensorDims{
              FFOrdered<positive_int>{4_p, 2_p},
          },
          DataType::FLOAT,
      };

      GenericTensorAccessorW result = mk_output(output_shape);

      CHECK_THROWS(copy_accessor_data_to_l_from_r_strict(result, input));
    }
  }
}
