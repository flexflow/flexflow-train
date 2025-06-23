#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/replicate_kernels_cpu.h"
#include "kernels/replicate_kernels_gpu.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Call Replicate Forward and Backward Kernels") {
    nonnegative_int num_replicas = 10_n;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{3_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{3_p}},
        DataType::FLOAT,
    };

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("gpu_forward_kernel") {
      GenericTensorAccessorR input =
          create_1d_accessor_r_with_contents<float>({1, 3, 2}, gpu_allocator);

      GenericTensorAccessorW output =
          gpu_allocator.allocate_tensor(output_shape);

      Kernels::Replicate::gpu_forward_kernel(
          managed_stream.raw_stream(), input, output);

      GenericTensorAccessorR correct = input;

      CHECK_MESSAGE(accessors_are_equal(output, correct),
                    check_kv("output", format_accessor_w_contents(output)));
    }

    SUBCASE("gpu_backward_kernel") {
      GenericTensorAccessorR output_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 2, 3},
                  {4, 3, 3},
                  {1, 3, 5},
              },
              gpu_allocator);

      GenericTensorAccessorR correct =
          create_1d_accessor_r_with_contents<float>(
              {1 + 2 + 3, 4 + 3 + 3, 1 + 3 + 5}, cpu_allocator);

      GenericTensorAccessorW input_grad =
          gpu_allocator.allocate_tensor(input_shape);

      Kernels::Replicate::gpu_backward_kernel(
          managed_stream.raw_stream(),
          output_grad,
          input_grad,
          num_replicas.unwrap_nonnegative());

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }

  TEST_CASE("Check Replicate Forward and Backward Kernel against CPU Kernel") {
    positive_int num_replicas = 2_p;

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{5_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{5_p, num_replicas}},
        DataType::FLOAT,
    };

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator gpu_allocator = create_local_cuda_memory_allocator();
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("gpu_forward_kernel") {
      // Run GPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_gpu =
          create_random_filled_accessor_r(input_shape, gpu_allocator);
      GenericTensorAccessorW output_accessor_gpu =
          create_zero_filled_accessor_w(output_shape, gpu_allocator);

      Kernels::Replicate::gpu_forward_kernel(
          managed_stream.raw_stream(), input_accessor_gpu, output_accessor_gpu);

      // Run CPU Replicate Forward Kernel
      GenericTensorAccessorR input_accessor_cpu =
          copy_tensor_accessor_r(input_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW output_accessor_cpu =
          create_zero_filled_accessor_w(output_shape, cpu_allocator);

      Kernels::Replicate::cpu_forward_kernel(input_accessor_cpu,
                                             output_accessor_cpu);

      CHECK_MESSAGE(
          accessors_are_equal(output_accessor_gpu, output_accessor_cpu),
          check_kv("input", format_accessor_r_contents(input_accessor_cpu)),
          check_kv("gpu", format_accessor_w_contents(output_accessor_gpu)),
          check_kv("cpu", format_accessor_w_contents(output_accessor_cpu)));
    }

    SUBCASE("gpu_backward_kernel") {
      // Run GPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_gpu =
          create_random_filled_accessor_r(output_shape, gpu_allocator);
      GenericTensorAccessorW input_grad_accessor_gpu =
          create_zero_filled_accessor_w(input_shape, gpu_allocator);

      Kernels::Replicate::gpu_backward_kernel(
          managed_stream.raw_stream(),
          output_grad_accessor_gpu,
          input_grad_accessor_gpu,
          num_replicas.int_from_positive_int());

      // Run CPU Replicate Backward Kernel
      GenericTensorAccessorR output_grad_accessor_cpu =
          copy_tensor_accessor_r(output_grad_accessor_gpu, cpu_allocator);
      GenericTensorAccessorW input_grad_accessor_cpu =
          create_zero_filled_accessor_w(input_shape, cpu_allocator);

      Kernels::Replicate::cpu_backward_kernel(
          output_grad_accessor_cpu,
          input_grad_accessor_cpu,
          num_replicas.int_from_positive_int());

      CHECK_MESSAGE(
          accessors_are_equal(input_grad_accessor_gpu, input_grad_accessor_cpu),
          check_kv("output_grad",
                   format_accessor_r_contents(output_grad_accessor_cpu)),
          check_kv("gpu", format_accessor_w_contents(input_grad_accessor_gpu)),
          check_kv("cpu", format_accessor_w_contents(input_grad_accessor_cpu)));
    }
  }
}
