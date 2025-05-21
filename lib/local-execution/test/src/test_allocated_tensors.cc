#include "local-execution/allocated_tensors.h"
#include "local-execution/gradient_tensor_source.h"
#include "kernels/local_cpu_allocator.h"
#include "local-execution/loss_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "pcg/computation_graph.dtg.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/variant.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test_utils.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("AllocatedTensors") {
    MockTensorGuidSource tensor_guid_source;
    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;
    LossTensorSource loss_tensor_source;

    Allocator allocator = create_local_cpu_memory_allocator();

    tensor_guid_t mock_tensor_1 = tensor_guid_source.new_mock_tensor_guid();
    tensor_guid_t mock_tensor_2 = tensor_guid_source.new_mock_tensor_guid();
    tensor_guid_t mock_tensor_3_with_grad =
        tensor_guid_source.new_mock_tensor_guid();
    tensor_guid_t dangling_tensor = tensor_guid_source.new_mock_tensor_guid();

    TensorAttrs tensor_attrs_1_no_grad = TensorAttrs{
        TensorShape{TensorDims{FFOrdered{16_p, 10_p}},
                    DataType::FLOAT},
        CreateGrad::NO};
    TensorAttrs tensor_attrs_2_no_grad = TensorAttrs{
        TensorShape{TensorDims{FFOrdered{16_p, 20_p}},
                    DataType::FLOAT},
        CreateGrad::NO};
    TensorAttrs tensor_attrs_3_with_grad = TensorAttrs{
        TensorShape{TensorDims{FFOrdered{16_p, 30_p}},
                    DataType::FLOAT},
        CreateGrad::YES};

    GenericTensorAccessorW tensor_backing_1 =
        allocator.allocate_tensor(tensor_attrs_1_no_grad.shape);
    GenericTensorAccessorW tensor_backing_2 =
        allocator.allocate_tensor(tensor_attrs_2_no_grad.shape);
    GenericTensorAccessorW tensor_backing_3 =
        allocator.allocate_tensor(tensor_attrs_3_with_grad.shape);

    std::unordered_map<tensor_guid_t, TensorAttrs> tensor_attrs_mapping = {
        {mock_tensor_1, tensor_attrs_1_no_grad},
        {mock_tensor_2, tensor_attrs_2_no_grad},
        {mock_tensor_3_with_grad, tensor_attrs_3_with_grad},
    };

    SUBCASE("Trivial tensors") {
      SUBCASE("Empty") {
        AllocatedTensors allocated_tensors = AllocatedTensors{{}, {}, {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == true);
      }

      SUBCASE("Loss tensor") {
        loss_tensor_t loss_tensor = loss_tensor_source.new_loss_tensor();
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{loss_tensor}, tensor_backing_1}}, {}, {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == true);
      }
    }

    SUBCASE("Forward tensors") {
      SUBCASE("Correct forward tensor") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{mock_tensor_1}, tensor_backing_1}}, {}, {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == true);
      }

      SUBCASE("Incorrect forward tensor") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{mock_tensor_1}, tensor_backing_2}}, {}, {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Dangling tensor guid") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {
                {TensorTypeVariant{dangling_tensor}, tensor_backing_1},
            },
            {},
            {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }
    }

    SUBCASE("Gradient tensors") {
      gradient_tensor_t grad_tensor_3 =
          gradient_tensor_source.new_gradient_tensor();

      SUBCASE("Gradient tensor") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{grad_tensor_3}, tensor_backing_3}},
            {{mock_tensor_3_with_grad, grad_tensor_3}},
            {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == true);
      }

      SUBCASE("Dangling gradient tensor") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{grad_tensor_3}, tensor_backing_3}}, {}, {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Dangling gradient tensor in mapping") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {}, {{mock_tensor_3_with_grad, grad_tensor_3}}, {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Gradient allocated for forward tensor without gradient") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{grad_tensor_3}, tensor_backing_3}},
            {{mock_tensor_2, grad_tensor_3}},
            {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Gradient tensor with wrong shape") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{grad_tensor_3}, tensor_backing_2}},
            {{mock_tensor_3_with_grad, grad_tensor_3}},
            {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Gradient tensor with dangling tensor guid") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{grad_tensor_3}, tensor_backing_3}},
            {{dangling_tensor, grad_tensor_3}},
            {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }
    }

    SUBCASE("Optimizer tensors") {
      optimizer_tensor_t optimizer_tensor_3 =
          optimizer_tensor_source.new_optimizer_tensor();

      SUBCASE("Optimizer tensor") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{optimizer_tensor_3}, tensor_backing_3}},
            {},
            {{mock_tensor_3_with_grad, {optimizer_tensor_3}}}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == true);
      }

      SUBCASE("Dangling optimizer tensor") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{optimizer_tensor_3}, tensor_backing_3}},
            {},
            {}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Dangling optimizer tensor in mapping") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {}, {}, {{mock_tensor_3_with_grad, {optimizer_tensor_3}}}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Optimizer allocated for forward tensor without gradient") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{optimizer_tensor_3}, tensor_backing_3}},
            {},
            {{mock_tensor_2, {optimizer_tensor_3}}}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Optimizer tensor with wrong shape") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{optimizer_tensor_3}, tensor_backing_2}},
            {},
            {{mock_tensor_3_with_grad, {optimizer_tensor_3}}}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }

      SUBCASE("Optimizer tensor with dangling tensor guid") {
        AllocatedTensors allocated_tensors = AllocatedTensors{
            {{TensorTypeVariant{optimizer_tensor_3}, tensor_backing_3}},
            {},
            {{dangling_tensor, {optimizer_tensor_3}}}};
        bool result = are_allocated_tensors_valid(allocated_tensors,
                                                  tensor_attrs_mapping);
        CHECK(result == false);
      }
    }
  }
}
