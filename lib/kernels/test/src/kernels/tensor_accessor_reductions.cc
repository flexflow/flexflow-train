#include <doctest/doctest.h>
#include "kernels/create_accessor_with_contents.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/tensor_accessor_reductions.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_accessor_all") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("returns false if any elements are false") {
      GenericTensorAccessorR accessor = create_3d_accessor_r_with_contents<bool>(
          {
            {
              {true, true, true},
              {true, true, true},
            },
            {
              {true, false, true},
              {true, true, true},
            },
          },
          cpu_allocator);

      bool result = tensor_accessor_all(accessor);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("returns true if all elements are true") {
      GenericTensorAccessorR accessor = create_2d_accessor_r_with_contents<bool>(
          {
            {true, true, true},
            {true, true, true},
          },
          cpu_allocator);

      bool result = tensor_accessor_all(accessor);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("throw an error if the datatype is not bool") {
      GenericTensorAccessorR accessor = create_2d_accessor_r_with_contents<int32_t>(
          {
            {1, 0, 1},
            {1, 1, 1},
          },
          cpu_allocator);

      CHECK_THROWS(tensor_accessor_all(accessor));
    }
  }

  TEST_CASE("tensor_accessor_any") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("returns true if any elements are true") {
      GenericTensorAccessorR accessor = create_3d_accessor_r_with_contents<bool>(
          {
            {
              {false, false, false},
              {true, false, false},
            },
            {
              {false, false, false},
              {false, false, false},
            },
          },
          cpu_allocator);

      bool result = tensor_accessor_any(accessor);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("returns false if all elements are false") {
      GenericTensorAccessorR accessor = create_2d_accessor_r_with_contents<bool>(
          {
            {false, false, false},
            {false, false, false},
          },
          cpu_allocator);

      bool result = tensor_accessor_any(accessor);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("throw an error if the datatype is not bool") {
      GenericTensorAccessorR accessor = create_2d_accessor_r_with_contents<int32_t>(
          {
            {1, 0, 1},
            {1, 1, 1},
          },
          cpu_allocator);

      CHECK_THROWS(tensor_accessor_any(accessor));
    }
  }
}
