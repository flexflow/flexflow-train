#include "kernels/map_tensor_accessors.h"
#include "kernels/create_accessor_with_contents.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("map_tensor_accessor_inplace") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW accessor = create_2d_accessor_w_with_contents<float>(
        {
            {1, 3, 2},
            {2, 1, 5},
        },
        cpu_allocator);

    map_tensor_accessor_inplace(accessor, [](float x) { return x + 1; });

    auto at = [&](nonnegative_int r, nonnegative_int c) -> float {
      return accessor.at<DataType::FLOAT>(FFOrdered{r, c});
    };

    CHECK(at(0_n, 0_n) == 2);
    CHECK(at(0_n, 1_n) == 4);
    CHECK(at(0_n, 2_n) == 3);
    CHECK(at(1_n, 0_n) == 3);
    CHECK(at(1_n, 1_n) == 2);
    CHECK(at(1_n, 2_n) == 6);
  }

  TEST_CASE("map_tensor_accessor") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW input = create_2d_accessor_w_with_contents<float>(
        {
            {1, 3, 2},
            {2, 1, 5},
        },
        cpu_allocator);

    SUBCASE("function is not type changing") {
      GenericTensorAccessorW result = map_tensor_accessor(
          input, [](float x) { return x + 1; }, cpu_allocator);

      auto at = [&](nonnegative_int r, nonnegative_int c) -> float {
        return result.at<DataType::FLOAT>(FFOrdered{r, c});
      };

      CHECK(at(0_n, 0_n) == 2);
      CHECK(at(0_n, 1_n) == 4);
      CHECK(at(0_n, 2_n) == 3);
      CHECK(at(1_n, 0_n) == 3);
      CHECK(at(1_n, 1_n) == 2);
      CHECK(at(1_n, 2_n) == 6);
    }

    SUBCASE("function is type changing") {
      GenericTensorAccessorW result = map_tensor_accessor(
          input, [](float x) -> bool { return x > 2; }, cpu_allocator);

      auto at = [&](nonnegative_int r, nonnegative_int c) -> bool {
        return result.at<DataType::BOOL>(FFOrdered{r, c});
      };

      CHECK(at(0_n, 0_n) == false);
      CHECK(at(0_n, 1_n) == true);
      CHECK(at(0_n, 2_n) == false);
      CHECK(at(1_n, 0_n) == false);
      CHECK(at(1_n, 1_n) == false);
      CHECK(at(1_n, 2_n) == true);
    }
  }

  TEST_CASE("map_tensor_accessors2") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    GenericTensorAccessorW lhs = create_2d_accessor_w_with_contents<float>(
        {
            {1, 3, 2},
            {2, 1, 5},
        },
        cpu_allocator);

    SUBCASE("argument types are the same") {
      GenericTensorAccessorW rhs = create_2d_accessor_w_with_contents<float>(
          {
              {0, 2, 5},
              {3, 3, 8},
          },
          cpu_allocator);

      SUBCASE("function is not type changing") {
        GenericTensorAccessorW result = map_tensor_accessors2(
            lhs,
            rhs,
            DataType::FLOAT,
            [](float l, float r) { return l + 2 * r; },
            cpu_allocator);

        auto at = [&](nonnegative_int r, nonnegative_int c) -> float {
          return result.at<DataType::FLOAT>(FFOrdered{r, c});
        };

        CHECK(at(0_n, 0_n) == 1);
        CHECK(at(0_n, 1_n) == 7);
        CHECK(at(0_n, 2_n) == 12);
        CHECK(at(1_n, 0_n) == 8);
        CHECK(at(1_n, 1_n) == 7);
        CHECK(at(1_n, 2_n) == 21);
      }

      SUBCASE("function is type changing") {
        GenericTensorAccessorW result = map_tensor_accessors2(
            lhs,
            rhs,
            DataType::BOOL,
            [](float l, float r) -> bool { return l > r; },
            cpu_allocator);

        auto at = [&](nonnegative_int r, nonnegative_int c) -> bool {
          return result.at<DataType::BOOL>(FFOrdered{r, c});
        };

        CHECK(at(0_n, 0_n) == true);
        CHECK(at(0_n, 1_n) == true);
        CHECK(at(0_n, 2_n) == false);
        CHECK(at(1_n, 0_n) == false);
        CHECK(at(1_n, 1_n) == false);
        CHECK(at(1_n, 2_n) == false);
      }
    }

    SUBCASE("argument types are not the same") {
      GenericTensorAccessorW rhs = create_2d_accessor_w_with_contents<bool>(
          {
              {true, false, true},
              {true, false, false},
          },
          cpu_allocator);

      auto func = [](float l, bool r) -> double {
        if (r) {
          return (-l);
        } else {
          return l * 2;
        }
      };
      GenericTensorAccessorW result = map_tensor_accessors2(
          lhs, rhs, DataType::DOUBLE, func, cpu_allocator);

      auto at = [&](nonnegative_int r, nonnegative_int c) -> double {
        return result.at<DataType::DOUBLE>(FFOrdered{r, c});
      };

      CHECK(at(0_n, 0_n) == -1);
      CHECK(at(0_n, 1_n) == 6);
      CHECK(at(0_n, 2_n) == -2);
      CHECK(at(1_n, 0_n) == -2);
      CHECK(at(1_n, 1_n) == 2);
      CHECK(at(1_n, 2_n) == 10);
    }
  }
}
