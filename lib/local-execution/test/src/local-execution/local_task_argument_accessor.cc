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

    std::unordered_map<tensor_sub_slot_id_t, TensorSlotBacking>
        tensor_slots_backing = {
            {
                tensor_sub_slot_id_t{slot_id_t{INPUT}, TensorType::FORWARD},
                TensorSlotBacking{input},
            },
            {
                tensor_sub_slot_id_t{slot_id_t{INPUT}, TensorType::GRADIENT},
                TensorSlotBacking{input_grad},
            },
            {
                tensor_sub_slot_id_t{slot_id_t{VARIADIC_TENSORS},
                                     TensorType::FORWARD},
                TensorSlotBacking{variadic_tensors},
            },
            {
                tensor_sub_slot_id_t{slot_id_t{VARIADIC_TENSORS},
                                     TensorType::GRADIENT},
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
      SUBCASE("get_tensor(slot_id_t, Permissions::RO, TensorType::FORWARD)") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input)};
        GenericTensorAccessor result = acc.get_tensor(
            slot_id_t{INPUT}, Permissions::RO, TensorType::FORWARD);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(slot_id_t, Permissions::RO, TensorType::GRADIENT)") {
        GenericTensorAccessor correct = GenericTensorAccessor{
            read_only_accessor_from_write_accessor(input_grad)};
        GenericTensorAccessor result = acc.get_tensor(
            slot_id_t{INPUT}, Permissions::RO, TensorType::GRADIENT);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(slot_id_t, Permissions::WO, TensorType::FORWARD)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result = acc.get_tensor(
            slot_id_t{INPUT}, Permissions::WO, TensorType::FORWARD);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(slot_id_t, Permissions::WO, TensorType::GRADIENT)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result = acc.get_tensor(
            slot_id_t{INPUT}, Permissions::WO, TensorType::GRADIENT);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(slot_id_t, Permissions::RW, TensorType::FORWARD)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input};
        GenericTensorAccessor result = acc.get_tensor(
            slot_id_t{INPUT}, Permissions::RW, TensorType::FORWARD);
        CHECK(correct == result);
      }

      SUBCASE("get_tensor(slot_id_t, Permissions::RW, TensorType::GRADIENT)") {
        GenericTensorAccessor correct = GenericTensorAccessor{input_grad};
        GenericTensorAccessor result = acc.get_tensor(
            slot_id_t{INPUT}, Permissions::RW, TensorType::GRADIENT);
        CHECK(correct == result);
      }
    }

    SUBCASE("get_variadic_tensor") {
      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::RO, "
              "TensorType::FORWARD)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{std::vector<GenericTensorAccessorR>{
                read_only_accessor_from_write_accessor(variadic_tensors.at(0)),
                read_only_accessor_from_write_accessor(
                    variadic_tensors.at(1))}};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RO, TensorType::FORWARD);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::RO, "
              "TensorType::GRADIENT)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{std::vector<GenericTensorAccessorR>{
                read_only_accessor_from_write_accessor(
                    variadic_tensors_grad.at(0)),
                read_only_accessor_from_write_accessor(
                    variadic_tensors_grad.at(1))}};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RO, TensorType::GRADIENT);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, "
              "TensorType::FORWARD)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::WO, TensorType::FORWARD);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, "
              "TensorType::GRADIENT)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors_grad};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::WO, TensorType::GRADIENT);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, "
              "TensorType::FORWARD)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RW, TensorType::FORWARD);
        CHECK(result == correct);
      }

      SUBCASE("get_variadic_tensor(slot_id_t, Permissions::WO, "
              "TensorType::GRADIENT)") {
        VariadicGenericTensorAccessor correct =
            VariadicGenericTensorAccessor{variadic_tensors_grad};
        VariadicGenericTensorAccessor result = acc.get_variadic_tensor(
            slot_id_t{VARIADIC_TENSORS}, Permissions::RW, TensorType::GRADIENT);
        CHECK(result == correct);
      }
    }
  }
}
