#include "internal/test_utils.h"
#include "kernels/combine_kernels_cpu.h"
#include "kernels/combine_kernels_gpu.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Call Combine Forward and Backward Kernels") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_p, 100_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Combine::gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Combine::gpu_backward_kernel(managed_stream.raw_stream(),
                                        output_grad_accessor,
                                        input_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }

  TEST_CASE("Check Combine Forward Kernel against CPU Kernel") {
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{5_p, 5_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      // Run GPU Combine Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          gpu_allocator.allocate_tensor(output_shape);

      Kernels::Combine::gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      // Run CPU Combine Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          cpu_allocator.allocate_tensor(output_shape);

      Kernels::Combine::cpu_forward_kernel(input_accessor_cpu,
                                           output_accessor_cpu);

      CHECK(accessors_are_equal(output_accessor_gpu, output_accessor_cpu));
    }

    SUBCASE("backward_kernel") {
      // Run GPU Combine Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r(output_shape, gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          create_zero_filled_accessor_w(input_shape, gpu_allocator);

      Kernels::Combine::gpu_backward_kernel(managed_stream.raw_stream(),
                                        output_grad_accessor_gpu,
                                        input_grad_accessor_gpu);

      // Run CPU Combine Backward Kernel
      GenericTensorAccessorR output_grad_accessor_cpu =
          copy_tensor_accessor_r(output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          create_zero_filled_accessor_w(input_shape, cpu_allocator);

      Kernels::Combine::cpu_backward_kernel(output_grad_accessor_cpu,
                                            input_grad_accessor_cpu);

      CHECK(accessors_are_equal(input_grad_accessor_gpu,
                                input_grad_accessor_cpu));
    }
  }
}
