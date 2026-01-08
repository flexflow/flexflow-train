#include "local-execution/local_task_argument_accessor.h"
#include "kernels/local_cpu_allocator.h"
#include "task-spec/task_signature_impl.h"
#include "utils/fmt/variant.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("LocalTaskArgumentAccessor") {
    Allocator allocator = create_local_cpu_memory_allocator();
    positive_int embed_dim = 32_p;
    positive_int num_heads = 10_p;

    positive_int batch_size = 40_p;
    positive_int seq_len = 48_p;
    positive_int feature_size = 36_p;

    DataType dtype = DataType::FLOAT;
    TensorShape input_tensor_shape = TensorShape{
        TensorDims{FFOrdered{batch_size, seq_len, feature_size}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW input =
        allocator.allocate_tensor(input_tensor_shape);
    GenericTensorAccessorW input_grad =
        allocator.allocate_tensor(input_tensor_shape);

    std::vector<GenericTensorAccessorW> variadic_tensors = {input, input};
    std::vector<GenericTensorAccessorW> variadic_tensors_grad = {input_grad,
                                                                 input_grad};

    enum Slots {
      INPUT,
      VARIADIC_TENSORS,
    };

    std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking>
        tensor_slots_backing = {
            {
                training_tensor_slot_id_t{TensorSlotName::LHS_INPUT,
                                          TrainingTensorType::FORWARD},
                TensorSlotBacking{input},
            },
            {
                training_tensor_slot_id_t{TensorSlotName::LHS_INPUT,
                                          TrainingTensorType::GRADIENT},
                TensorSlotBacking{input_grad},
            },
            {
                training_tensor_slot_id_t{TensorSlotName::INPUT,
                                          TrainingTensorType::FORWARD},
                TensorSlotBacking{variadic_tensors},
            },
            {
                training_tensor_slot_id_t{TensorSlotName::INPUT,
                                          TrainingTensorType::GRADIENT},
                TensorSlotBacking{variadic_tensors_grad},
            },
        };

    LocalTaskArgumentAccessor acc = LocalTaskArgumentAccessor{
        /*allocator=*/allocator,
        /*tensor_slots_backing=*/tensor_slots_backing,
        /*arg_slots_backing=*/{},
        /*device_idx=*/0,
    };

    SUBCASE("get_tensor") {
      SUBCASE("get_tensor(TensorSlotName, Permissions::RO, "
              "TrainingTensorType::FORWARD)") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input)};
        GenericTensorAccessor result =
            acc.get_tensor(TensorSlotName::LHS_INPUT,
                           Permissions::RO,
                           TrainingTensorType::FORWARD);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(TensorSlotName, Permissions::RO, "
              "TrainingTensorType::GRADIENT)") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input_grad)};
        GenericTensorAccessor result =
            acc.get_tensor(TensorSlotName::LHS_INPUT,
                           Permissions::RO,
                           TrainingTensorType::GRADIENT);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(TensorSlotName, Permissions::WO, "
              "TrainingTensorType::FORWARD)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result =
            acc.get_tensor(TensorSlotName::LHS_INPUT,
                           Permissions::WO,
                           TrainingTensorType::FORWARD);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(TensorSlotName, Permissions::WO, "
              "TrainingTensorType::GRADIENT)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result =
            acc.get_tensor(TensorSlotName::LHS_INPUT,
                           Permissions::WO,
                           TrainingTensorType::GRADIENT);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(TensorSlotName, Permissions::RW, "
              "TrainingTensorType::FORWARD)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result =
            acc.get_tensor(TensorSlotName::LHS_INPUT,
                           Permissions::RW,
                           TrainingTensorType::FORWARD);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(TensorSlotName, Permissions::RW, "
              "TrainingTensorType::GRADIENT)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result =
            acc.get_tensor(TensorSlotName::LHS_INPUT,
                           Permissions::RW,
                           TrainingTensorType::GRADIENT);
        CHECK(correct == result);
      }
    }

    SUBCASE("get_variadic_tensor") {
      SUBCASE("get_variadic_tensor(TensorSlotName, Permissions::RO, "
              "TrainingTensorType::FORWARD)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{std::vector<GenericTensorAccessorR>{
                read_only_accessor_from_write_accessor(variadic_tensors.at(0)),
                read_only_accessor_from_write_accessor(
                    variadic_tensors.at(1))}};
        VariadicGenericTensorAccessor result =
            acc.get_variadic_tensor(TensorSlotName::INPUT,
                                    Permissions::RO,
                                    TrainingTensorType::FORWARD);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(TensorSlotName, Permissions::RO, "
              "TrainingTensorType::GRADIENT)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{std::vector<GenericTensorAccessorR>{
                read_only_accessor_from_write_accessor(
                    variadic_tensors_grad.at(0)),
                read_only_accessor_from_write_accessor(
                    variadic_tensors_grad.at(1))}};
        VariadicGenericTensorAccessor result =
            acc.get_variadic_tensor(TensorSlotName::INPUT,
                                    Permissions::RO,
                                    TrainingTensorType::GRADIENT);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(TensorSlotName, Permissions::WO, "
              "TrainingTensorType::FORWARD)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors};
        VariadicGenericTensorAccessor result =
            acc.get_variadic_tensor(TensorSlotName::INPUT,
                                    Permissions::WO,
                                    TrainingTensorType::FORWARD);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(TensorSlotName, Permissions::WO, "
              "TrainingTensorType::GRADIENT)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors_grad};
        VariadicGenericTensorAccessor result =
            acc.get_variadic_tensor(TensorSlotName::INPUT,
                                    Permissions::WO,
                                    TrainingTensorType::GRADIENT);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(TensorSlotName, Permissions::WO, "
              "TrainingTensorType::FORWARD)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors};
        VariadicGenericTensorAccessor result =
            acc.get_variadic_tensor(TensorSlotName::INPUT,
                                    Permissions::RW,
                                    TrainingTensorType::FORWARD);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(TensorSlotName, Permissions::WO, "
              "TrainingTensorType::GRADIENT)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors_grad};
        VariadicGenericTensorAccessor result =
            acc.get_variadic_tensor(TensorSlotName::INPUT,
                                    Permissions::RW,
                                    TrainingTensorType::GRADIENT);
        CHECK(result == correct);
      }
    }
  }
}
