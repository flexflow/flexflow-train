#include "doctest/doctest.h"
#include "kernels/accessor.h"
#include "op-attrs/datatype_value.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test copy_tensor_accessor") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    TensorShape shape =
        make_tensor_shape_from_ff_ordered({5_n, 5_n}, DataType::FLOAT);

    SUBCASE("Test copy_tensor_accessor_r") {
      GenericTensorAccessorR src_accessor =
          create_random_filled_accessor_r(shape, cpu_allocator);
      GenericTensorAccessorR dst_accessor =
          copy_tensor_accessor_r(src_accessor, cpu_allocator);

      CHECK(accessor_data_is_equal(src_accessor, dst_accessor));
    }

    SUBCASE("Test copy_tensor_accessor_w") {
      GenericTensorAccessorW src_accessor =
          create_random_filled_accessor_w(shape, cpu_allocator);
      GenericTensorAccessorW dst_accessor =
          copy_tensor_accessor_w(src_accessor, cpu_allocator);

      CHECK(accessor_data_is_equal(src_accessor, dst_accessor));
    }

    SUBCASE("Test copy_accessor_r_to_cpu_if_necessary") {
      SUBCASE("Test necessary") {
        GenericTensorAccessorR src_accessor =
            create_random_filled_accessor_r(shape, gpu_allocator);
        GenericTensorAccessorR dst_accessor =
            copy_accessor_r_to_cpu_if_necessary(src_accessor, cpu_allocator);

        CHECK(accessor_data_is_equal(src_accessor, dst_accessor));
        CHECK(dst_accessor.device_type == DeviceType::CPU);
      }

      SUBCASE("Test not necessary") {
        GenericTensorAccessorR src_accessor =
            create_random_filled_accessor_r(shape, cpu_allocator);
        GenericTensorAccessorR dst_accessor =
            copy_accessor_r_to_cpu_if_necessary(src_accessor, cpu_allocator);

        CHECK(accessor_data_is_equal(src_accessor, dst_accessor));
        CHECK(dst_accessor.device_type == DeviceType::CPU);
      }
    }

    SUBCASE("Test copy_accessor_w_to_cpu_if_necessary") {
      SUBCASE("Test necessary") {
        GenericTensorAccessorW src_accessor =
            create_random_filled_accessor_w(shape, gpu_allocator);
        GenericTensorAccessorW dst_accessor =
            copy_accessor_w_to_cpu_if_necessary(src_accessor, cpu_allocator);

        CHECK(accessor_data_is_equal(src_accessor, dst_accessor));
        CHECK(dst_accessor.device_type == DeviceType::CPU);
      }

      SUBCASE("Test not necessary") {
        GenericTensorAccessorW src_accessor =
            create_random_filled_accessor_w(shape, cpu_allocator);
        GenericTensorAccessorW dst_accessor =
            copy_accessor_w_to_cpu_if_necessary(src_accessor, cpu_allocator);

        CHECK(accessor_data_is_equal(src_accessor, dst_accessor));
        CHECK(dst_accessor.device_type == DeviceType::CPU);
      }
    }
  }
}
