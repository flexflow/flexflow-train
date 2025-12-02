#include "local-execution/local_tensor_backing.h"
#include "internal/test_utils.h"
#include "kernels/local_cpu_allocator.h"
#include "task-spec/gradient_tensor_source.h"
#include "task-spec/loss_tensor_source.h"
#include "task-spec/optimizer_tensor_source.h"
#include "test/utils/doctest/check_kv.h"
#include "test/utils/doctest/fmt/unordered_map.h"
#include "utils/containers/keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

bool is_shape_and_dtype_equal_for_tensor_backings(
    LocalTensorBacking const &b1, LocalTensorBacking const &b2) {

  std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> m1 =
      b1.backing_for_training_tensor_map;
  std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> m2 =
      b2.backing_for_training_tensor_map;

  if (keys(m1) == keys(m2)) {
    for (std::pair<training_tensor_guid_t, GenericTensorAccessorW> const
             &tensor_type_backing : m1) {
      if (tensor_type_backing.second.shape ==
          m2.at(tensor_type_backing.first).shape) {
        continue;
      } else {
        return false;
      }
    }
    return true;
  } else {
    return false;
  }
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("construct_local_tensor_backing") {
    Allocator allocator = create_local_cpu_memory_allocator();

    training_tensor_guid_t t1 =
        training_tensor_guid_t{forward_tensor_guid_t{4}};
    training_tensor_guid_t t2 =
        training_tensor_guid_t{gradient_tensor_guid_t{4}};
    training_tensor_guid_t t3 =
        training_tensor_guid_t{gradient_tensor_guid_t{5}};
    training_tensor_guid_t t4 =
        training_tensor_guid_t{gradient_tensor_guid_t{6}};

    TensorShape tensor_shape_1 = TensorShape{
        TensorDims{FFOrdered{
            4_p,
            5_p,
        }},
        DataType::FLOAT,
    };

    TensorShape tensor_shape_2 = TensorShape{
        TensorDims{FFOrdered{
            4_p,
            5_p,
        }},
        DataType::FLOAT,
    };

    std::unordered_map<training_tensor_guid_t, TensorShape>
        training_tensor_shapes = {
            {t1, tensor_shape_1},
            {t2, tensor_shape_2},
            {t3, tensor_shape_1},
        };

    GenericTensorAccessorW t3_accessor =
        allocator.allocate_tensor(tensor_shape_2);
    SUBCASE("allocates all non-preallocated tensors and does not re-allocate "
            "the preallocated ones") {
      std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW>
          preallocated_tensors = {
              {t3, t3_accessor},
          };

      LocalTensorBacking result = construct_local_tensor_backing(
          /*training_tensor_shapes=*/training_tensor_shapes,
          /*preallocated_tensors=*/preallocated_tensors,
          /*allocator=*/allocator);
      LocalTensorBacking correct = LocalTensorBacking{
          /*backing_for_training_tensor_map=*/{
              {t3, t3_accessor},
              {t1, allocator.allocate_tensor(tensor_shape_1)},
              {t2, allocator.allocate_tensor(tensor_shape_2)},
          },
      };

      CHECK_MESSAGE(
          is_shape_and_dtype_equal_for_tensor_backings(result, correct),
          check_kv("result", fmt::to_string(result)),
          check_kv("correct", fmt::to_string(correct)));

      CHECK(get_accessor_for_training_tensor(result, t3) == t3_accessor);
    }

    SUBCASE("fails if a preallocated tensor is not in training_tensor_shapes") {
      std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW>
          preallocated_tensors = {
              {t4, t3_accessor},
          };

      CHECK_THROWS(construct_local_tensor_backing(
          /*training_tensor_shapes=*/training_tensor_shapes,
          /*preallocated_tensors=*/preallocated_tensors,
          /*allocator=*/allocator));
    }
  }

  TEST_CASE("get_accessor_for_training_tensor") {
    Allocator allocator = create_local_cpu_memory_allocator();

    TensorShape tensor_shape = TensorShape{
        TensorDims{FFOrdered{
            4_p,
            5_p,
        }},
        DataType::FLOAT,
    };

    training_tensor_guid_t t1 =
        training_tensor_guid_t{forward_tensor_guid_t{4}};
    training_tensor_guid_t t2 =
        training_tensor_guid_t{gradient_tensor_guid_t{4}};

    GenericTensorAccessorW t1_accessor =
        allocator.allocate_tensor(tensor_shape);
    GenericTensorAccessorW t2_accessor =
        allocator.allocate_tensor(tensor_shape);

    LocalTensorBacking local_tensor_backing = LocalTensorBacking{
        /*backing_for_training_tensor_map=*/{{
                                                 t1,
                                                 t1_accessor,
                                             },
                                             {
                                                 t2,
                                                 t2_accessor,
                                             }},
    };

    SUBCASE("returns corresponding accessor if training tensor is present") {
      GenericTensorAccessorW result =
          get_accessor_for_training_tensor(local_tensor_backing, t1);
      GenericTensorAccessorW correct = t1_accessor;

      CHECK(result == correct);
    }

    SUBCASE("fails if the training tensor is not present") {
      training_tensor_guid_t t3 =
          training_tensor_guid_t{optimizer_tensor_guid_t{4}};
      training_tensor_guid_t t4 =
          training_tensor_guid_t{forward_tensor_guid_t{3}};

      CHECK_THROWS(get_accessor_for_training_tensor(local_tensor_backing, t3));
      CHECK_THROWS(get_accessor_for_training_tensor(local_tensor_backing, t4));
    }
  }

  TEST_CASE("construct_tensor_slots_backing_for_binding") {
    enum Slots {
      TENSOR_SLOT_1,
      TENSOR_SLOT_2,
      TENSOR_SLOT_3,
      ARG_SLOT,
    };

    Allocator allocator = create_local_cpu_memory_allocator();

    TensorShape tensor_shape = TensorShape{
        TensorDims{FFOrdered{
            4_p,
            5_p,
        }},
        DataType::FLOAT,
    };

    training_tensor_guid_t t1 =
        training_tensor_guid_t{forward_tensor_guid_t{4}};
    training_tensor_guid_t t2 =
        training_tensor_guid_t{forward_tensor_guid_t{5}};
    training_tensor_guid_t t3 =
        training_tensor_guid_t{forward_tensor_guid_t{6}};
    training_tensor_guid_t t4 =
        training_tensor_guid_t{gradient_tensor_guid_t{5}};

    GenericTensorAccessorW t1_accessor =
        allocator.allocate_tensor(tensor_shape);
    GenericTensorAccessorW t2_accessor =
        allocator.allocate_tensor(tensor_shape);
    GenericTensorAccessorW t3_accessor =
        allocator.allocate_tensor(tensor_shape);
    GenericTensorAccessorW t4_accessor =
        allocator.allocate_tensor(tensor_shape);

    training_tensor_slot_id_t tensor_slot_1_forward = training_tensor_slot_id_t{
        TensorSlotName::QUERY,
        TrainingTensorType::FORWARD,
    };
    training_tensor_slot_id_t tensor_slot_1_gradient = training_tensor_slot_id_t{
        TensorSlotName::QUERY,
        TrainingTensorType::GRADIENT,
    };
    training_tensor_slot_id_t tensor_slot_2_forward = training_tensor_slot_id_t{
        TensorSlotName::KEY,
        TrainingTensorType::FORWARD,
    };
    training_tensor_slot_id_t tensor_slot_3_forward = training_tensor_slot_id_t{
        TensorSlotName::VALUE,
        TrainingTensorType::FORWARD,
    };

    LocalTensorBacking local_tensor_backing = LocalTensorBacking{
        /*backing_for_training_tensor_map=*/{{
                                                 t1,
                                                 t1_accessor,
                                             },
                                             {
                                                 t2,
                                                 t2_accessor,
                                             },
                                             {
                                                 t3,
                                                 t3_accessor,
                                             },
                                             {
                                                 t4,
                                                 t4_accessor,
                                             }},
    };

    TaskBinding task_binding = TaskBinding{
        /*tensor_bindings=*/{
            {
                tensor_slot_1_forward,
                t1,
            },
            {
                tensor_slot_2_forward,
                t2,
            },
            {
                tensor_slot_1_gradient,
                t4,
            },
        },
        /*arg_bindings=*/
        {
            {
                arg_slot_id_t{ARG_SLOT},
                TaskArgSpec{
                    ConcreteArgSpec::create<int>(4),
                },
            },
        },
    };

    std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking> result =
        construct_tensor_slots_backing_for_binding(local_tensor_backing,
                                                   task_binding);
    std::unordered_map<training_tensor_slot_id_t, TensorSlotBacking> correct = {
        {
            tensor_slot_1_forward,
            TensorSlotBacking{t1_accessor},
        },
        {
            tensor_slot_2_forward,
            TensorSlotBacking{t2_accessor},
        },
        {
            tensor_slot_1_gradient,
            TensorSlotBacking{t4_accessor},
        },
    };

    CHECK(result == correct);
  }
}
