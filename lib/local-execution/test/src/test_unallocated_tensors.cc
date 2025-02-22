#include "local-execution/allocated_tensors.h"
#include "local-execution/gradient_tensor_source.h"
#include "local-execution/local_cpu_allocator.h"
#include "local-execution/loss_tensor_source.h"
#include "local-execution/optimizer_tensor_source.h"
#include "local-execution/unallocated_tensors.h"
#include "pcg/computation_graph.dtg.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "test/utils/doctest/fmt/variant.h"
#include "test/utils/doctest/fmt/vector.h"
#include "test_utils.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("UnallocatedTensors") {
    MockTensorGuidSource tensor_guid_source;
    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;

    gradient_tensor_source.reset();
    optimizer_tensor_source.reset();

    Allocator allocator = create_local_cpu_memory_allocator();

    tensor_guid_t mock_tensor_1 = tensor_guid_source.new_mock_tensor_guid();
    tensor_guid_t mock_tensor_2 = tensor_guid_source.new_mock_tensor_guid();
    tensor_guid_t mock_tensor_3_with_grad =
        tensor_guid_source.new_mock_tensor_guid();

    gradient_tensor_t grad_tensor =
        gradient_tensor_source.new_gradient_tensor();
    optimizer_tensor_t optimizer_tensor_1 =
        optimizer_tensor_source.new_optimizer_tensor();
    optimizer_tensor_t optimizer_tensor_2 =
        optimizer_tensor_source.new_optimizer_tensor();

    TensorAttrs tensor_attrs_1_no_grad = TensorAttrs{
        TensorShape{TensorDims{FFOrdered<nonnegative_int>{16_n, 10_n}},
                    DataType::FLOAT},
        std::nullopt,
        std::nullopt,
        CreateGrad::NO};
    TensorAttrs tensor_attrs_2_no_grad = TensorAttrs{
        TensorShape{TensorDims{FFOrdered<nonnegative_int>{16_n, 20_n}},
                    DataType::FLOAT},
        std::nullopt,
        std::nullopt,
        CreateGrad::NO};
    TensorAttrs tensor_attrs_3_with_grad = TensorAttrs{
        TensorShape{TensorDims{FFOrdered<nonnegative_int>{16_n, 30_n}},
                    DataType::FLOAT},
        std::nullopt,
        std::nullopt,
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

    SUBCASE("Without optimizer") {
      SUBCASE("AllocatedTensors is empty") {
        AllocatedTensors empty = AllocatedTensors{{}, {}, {}};
        gradient_tensor_source.reset();
        UnallocatedTensors result = generate_unallocated_tensors(
            empty, tensor_attrs_mapping, gradient_tensor_source);

        std::unordered_map<TensorTypeVariant, TensorShape>
            correct_tensor_type_shapes = {
                {TensorTypeVariant{mock_tensor_1},
                 tensor_attrs_1_no_grad.shape},
                {TensorTypeVariant{mock_tensor_2},
                 tensor_attrs_2_no_grad.shape},
                {TensorTypeVariant{mock_tensor_3_with_grad},
                 tensor_attrs_3_with_grad.shape},
                {TensorTypeVariant{grad_tensor},
                 tensor_attrs_3_with_grad.shape},
            };
        UnallocatedTensors correct =
            UnallocatedTensors{correct_tensor_type_shapes,
                               {{mock_tensor_3_with_grad, grad_tensor}},
                               {}};
        CHECK(result == correct);
      }

      SUBCASE("AllocatedTensors contains only 1 forward tensor") {
        AllocatedTensors allocated_forward_tensors = AllocatedTensors{
            {
                {TensorTypeVariant{mock_tensor_1}, tensor_backing_1},
            },
            {},
            {}};

        gradient_tensor_source.reset();
        UnallocatedTensors result =
            generate_unallocated_tensors(allocated_forward_tensors,
                                         tensor_attrs_mapping,
                                         gradient_tensor_source);
        std::unordered_map<TensorTypeVariant, TensorShape>
            correct_tensor_type_shapes = {
                {TensorTypeVariant{mock_tensor_2},
                 tensor_attrs_2_no_grad.shape},
                {TensorTypeVariant{mock_tensor_3_with_grad},
                 tensor_attrs_3_with_grad.shape},
                {TensorTypeVariant{grad_tensor},
                 tensor_attrs_3_with_grad.shape},
            };
        UnallocatedTensors correct =
            UnallocatedTensors{correct_tensor_type_shapes,
                               {{mock_tensor_3_with_grad, grad_tensor}},
                               {}};
        CHECK(result == correct);
      }

      SUBCASE("AllocatedTensors contains only forward tensors") {
        AllocatedTensors allocated_forward_tensors = AllocatedTensors{
            {
                {TensorTypeVariant{mock_tensor_1}, tensor_backing_1},
                {TensorTypeVariant{mock_tensor_2}, tensor_backing_2},
                {TensorTypeVariant{mock_tensor_3_with_grad}, tensor_backing_3},
            },
            {},
            {}};

        gradient_tensor_source.reset();
        UnallocatedTensors result =
            generate_unallocated_tensors(allocated_forward_tensors,
                                         tensor_attrs_mapping,
                                         gradient_tensor_source);

        std::unordered_map<TensorTypeVariant, TensorShape>
            correct_tensor_type_shapes = {
                {TensorTypeVariant{grad_tensor},
                 tensor_attrs_3_with_grad.shape},
            };
        UnallocatedTensors correct =
            UnallocatedTensors{correct_tensor_type_shapes,
                               {{mock_tensor_3_with_grad, grad_tensor}},
                               {}};
        CHECK(result == correct);
      }

      SUBCASE("AllocatedTensors contains only gradient tensor") {

        AllocatedTensors allocated_forward_tensors = AllocatedTensors{
            {
                {TensorTypeVariant{grad_tensor}, tensor_backing_3},
            },
            {{mock_tensor_3_with_grad, grad_tensor}},
            {}};
        UnallocatedTensors result =
            generate_unallocated_tensors(allocated_forward_tensors,
                                         tensor_attrs_mapping,
                                         gradient_tensor_source);

        std::unordered_map<TensorTypeVariant, TensorShape>
            correct_tensor_type_shapes = {
                {TensorTypeVariant{mock_tensor_1},
                 tensor_attrs_1_no_grad.shape},
                {TensorTypeVariant{mock_tensor_2},
                 tensor_attrs_2_no_grad.shape},
                {TensorTypeVariant{mock_tensor_3_with_grad},
                 tensor_attrs_3_with_grad.shape},
            };
        UnallocatedTensors correct =
            UnallocatedTensors{correct_tensor_type_shapes, {}, {}};
        CHECK(result == correct);
      }

      SUBCASE("AllocatedTensors contains mixture") {

        AllocatedTensors allocated_forward_tensors = AllocatedTensors{
            {
                {TensorTypeVariant{mock_tensor_1}, tensor_backing_1},
                {TensorTypeVariant{grad_tensor}, tensor_backing_3},
            },
            {{mock_tensor_3_with_grad, grad_tensor}},
            {}};
        UnallocatedTensors result =
            generate_unallocated_tensors(allocated_forward_tensors,
                                         tensor_attrs_mapping,
                                         gradient_tensor_source);

        std::unordered_map<TensorTypeVariant, TensorShape>
            correct_tensor_type_shapes = {
                {TensorTypeVariant{mock_tensor_2},
                 tensor_attrs_2_no_grad.shape},
                {TensorTypeVariant{mock_tensor_3_with_grad},
                 tensor_attrs_3_with_grad.shape},
            };
        UnallocatedTensors correct =
            UnallocatedTensors{correct_tensor_type_shapes, {}, {}};
        CHECK(result == correct);
      }

      SUBCASE("Fully AllocatedTensors") {

        AllocatedTensors allocated_forward_tensors = AllocatedTensors{
            {
                {TensorTypeVariant{mock_tensor_1}, tensor_backing_1},
                {TensorTypeVariant{mock_tensor_2}, tensor_backing_2},
                {TensorTypeVariant{mock_tensor_3_with_grad}, tensor_backing_3},
                {TensorTypeVariant{grad_tensor}, tensor_backing_3},
            },
            {{mock_tensor_3_with_grad, grad_tensor}},
            {}};
        UnallocatedTensors result =
            generate_unallocated_tensors(allocated_forward_tensors,
                                         tensor_attrs_mapping,
                                         gradient_tensor_source);

        UnallocatedTensors correct = UnallocatedTensors{{}, {}, {}};
        CHECK(result == correct);
      }
    }

    SUBCASE("With optimizer") {
      SUBCASE("SGD Attrs") {
        SUBCASE("without momentum") {
          double momentum = 0.0;
          OptimizerAttrs attrs =
              OptimizerAttrs{SGDOptimizerAttrs{0.0, momentum, false, 0.0}};
          AllocatedTensors empty = AllocatedTensors{{}, {}, {}};

          gradient_tensor_source.reset();
          UnallocatedTensors result =
              generate_unallocated_tensors_with_optimizer(
                  empty,
                  tensor_attrs_mapping,
                  gradient_tensor_source,
                  optimizer_tensor_source,
                  attrs);

          gradient_tensor_source.reset();
          UnallocatedTensors correct = generate_unallocated_tensors(
              empty, tensor_attrs_mapping, gradient_tensor_source);
          CHECK(result == correct);
        }
        SUBCASE("with momentum") {
          double momentum = 0.9;
          OptimizerAttrs attrs =
              OptimizerAttrs{SGDOptimizerAttrs{0.0, momentum, false, 0.0}};

          SUBCASE("unallocated") {
            AllocatedTensors empty = AllocatedTensors{{}, {}, {}};

            gradient_tensor_source.reset();
            optimizer_tensor_source.reset();
            UnallocatedTensors result =
                generate_unallocated_tensors_with_optimizer(
                    empty,
                    tensor_attrs_mapping,
                    gradient_tensor_source,
                    optimizer_tensor_source,
                    attrs);

            std::unordered_map<TensorTypeVariant, TensorShape>
                correct_tensor_type_shapes = {
                    {TensorTypeVariant{mock_tensor_1},
                     tensor_attrs_1_no_grad.shape},
                    {TensorTypeVariant{mock_tensor_2},
                     tensor_attrs_2_no_grad.shape},
                    {TensorTypeVariant{mock_tensor_3_with_grad},
                     tensor_attrs_3_with_grad.shape},
                    {TensorTypeVariant{grad_tensor},
                     tensor_attrs_3_with_grad.shape},
                    {TensorTypeVariant{optimizer_tensor_1},
                     tensor_attrs_3_with_grad.shape},
                };
            UnallocatedTensors correct = UnallocatedTensors{
                correct_tensor_type_shapes,
                {{mock_tensor_3_with_grad, grad_tensor}},
                {{mock_tensor_3_with_grad, {optimizer_tensor_1}}}};

            CHECK(result == correct);
          }

          SUBCASE("allocated") {

            AllocatedTensors allocated_optimizer_tensor = AllocatedTensors{
                {{TensorTypeVariant{optimizer_tensor_1}, tensor_backing_3}},
                {},
                {{mock_tensor_3_with_grad, {optimizer_tensor_1}}}};

            gradient_tensor_source.reset();
            UnallocatedTensors result =
                generate_unallocated_tensors_with_optimizer(
                    allocated_optimizer_tensor,
                    tensor_attrs_mapping,
                    gradient_tensor_source,
                    optimizer_tensor_source,
                    attrs);

            std::unordered_map<TensorTypeVariant, TensorShape>
                correct_tensor_type_shapes = {
                    {TensorTypeVariant{mock_tensor_1},
                     tensor_attrs_1_no_grad.shape},
                    {TensorTypeVariant{mock_tensor_2},
                     tensor_attrs_2_no_grad.shape},
                    {TensorTypeVariant{mock_tensor_3_with_grad},
                     tensor_attrs_3_with_grad.shape},
                    {TensorTypeVariant{grad_tensor},
                     tensor_attrs_3_with_grad.shape},
                };
            UnallocatedTensors correct =
                UnallocatedTensors{correct_tensor_type_shapes,
                                   {{mock_tensor_3_with_grad, grad_tensor}},
                                   {}};

            CHECK(result == correct);
          }
        }
      }
      SUBCASE("Adam Attrs") {
        OptimizerAttrs attrs =
            OptimizerAttrs{AdamOptimizerAttrs{/*alpha=*/0.001,
                                              /*beta1=*/0.9,
                                              /*beta2=*/0.999,
                                              /*weight_decay=*/0.001,
                                              /*alpha_t=*/0.001,
                                              /*beta_t=*/0.9,
                                              /*beta2_t=*/0.999,
                                              /*epsilon=*/1e-8}};
        SUBCASE("Empty") {
          AllocatedTensors empty = AllocatedTensors{{}, {}, {}};

          gradient_tensor_source.reset();
          optimizer_tensor_source.reset();
          UnallocatedTensors result =
              generate_unallocated_tensors_with_optimizer(
                  empty,
                  tensor_attrs_mapping,
                  gradient_tensor_source,
                  optimizer_tensor_source,
                  attrs);

          std::unordered_map<TensorTypeVariant, TensorShape>
              correct_tensor_type_shapes = {
                  {TensorTypeVariant{mock_tensor_1},
                   tensor_attrs_1_no_grad.shape},
                  {TensorTypeVariant{mock_tensor_2},
                   tensor_attrs_2_no_grad.shape},
                  {TensorTypeVariant{mock_tensor_3_with_grad},
                   tensor_attrs_3_with_grad.shape},
                  {TensorTypeVariant{grad_tensor},
                   tensor_attrs_3_with_grad.shape},
                  {TensorTypeVariant{optimizer_tensor_1},
                   tensor_attrs_3_with_grad.shape},
                  {TensorTypeVariant{optimizer_tensor_2},
                   tensor_attrs_3_with_grad.shape},
              };
          UnallocatedTensors correct =
              UnallocatedTensors{correct_tensor_type_shapes,
                                 {{mock_tensor_3_with_grad, grad_tensor}},
                                 {{mock_tensor_3_with_grad,
                                   {optimizer_tensor_1, optimizer_tensor_2}}}};

          CHECK(result == correct);
        }
        SUBCASE("Partially allocated") {
          gradient_tensor_source.reset();
          optimizer_tensor_source.reset();
          optimizer_tensor_t optimizer_tensor_pre_allocated =
              optimizer_tensor_source.new_optimizer_tensor();
          AllocatedTensors allocated_optimizer_tensor = AllocatedTensors{
              {{TensorTypeVariant{optimizer_tensor_pre_allocated},
                tensor_backing_3}},
              {},
              {{mock_tensor_3_with_grad, {optimizer_tensor_pre_allocated}}}};

          UnallocatedTensors result =
              generate_unallocated_tensors_with_optimizer(
                  allocated_optimizer_tensor,
                  tensor_attrs_mapping,
                  gradient_tensor_source,
                  optimizer_tensor_source,
                  attrs);

          std::unordered_map<TensorTypeVariant, TensorShape>
              correct_tensor_type_shapes = {
                  {TensorTypeVariant{mock_tensor_1},
                   tensor_attrs_1_no_grad.shape},
                  {TensorTypeVariant{mock_tensor_2},
                   tensor_attrs_2_no_grad.shape},
                  {TensorTypeVariant{mock_tensor_3_with_grad},
                   tensor_attrs_3_with_grad.shape},
                  {TensorTypeVariant{grad_tensor},
                   tensor_attrs_3_with_grad.shape},
                  {TensorTypeVariant{optimizer_tensor_2},
                   tensor_attrs_3_with_grad.shape},
              };
          UnallocatedTensors correct = UnallocatedTensors{
              correct_tensor_type_shapes,
              {{mock_tensor_3_with_grad, grad_tensor}},
              {{mock_tensor_3_with_grad, {optimizer_tensor_2}}}};

          CHECK(result == correct);
        }

        SUBCASE("Fully allocated") {
          AllocatedTensors allocated_optimizer_tensor = AllocatedTensors{
              {{TensorTypeVariant{optimizer_tensor_1}, tensor_backing_3},
               {TensorTypeVariant{optimizer_tensor_2}, tensor_backing_3}},
              {},
              {{mock_tensor_3_with_grad,
                {optimizer_tensor_1, optimizer_tensor_2}}}};

          gradient_tensor_source.reset();
          UnallocatedTensors result =
              generate_unallocated_tensors_with_optimizer(
                  allocated_optimizer_tensor,
                  tensor_attrs_mapping,
                  gradient_tensor_source,
                  optimizer_tensor_source,
                  attrs);

          std::unordered_map<TensorTypeVariant, TensorShape>
              correct_tensor_type_shapes = {
                  {TensorTypeVariant{mock_tensor_1},
                   tensor_attrs_1_no_grad.shape},
                  {TensorTypeVariant{mock_tensor_2},
                   tensor_attrs_2_no_grad.shape},
                  {TensorTypeVariant{mock_tensor_3_with_grad},
                   tensor_attrs_3_with_grad.shape},
                  {TensorTypeVariant{grad_tensor},
                   tensor_attrs_3_with_grad.shape},
              };
          UnallocatedTensors correct =
              UnallocatedTensors{correct_tensor_type_shapes,
                                 {{mock_tensor_3_with_grad, grad_tensor}},
                                 {}};

          CHECK(result == correct);
        }
      }
    }
  }
}
