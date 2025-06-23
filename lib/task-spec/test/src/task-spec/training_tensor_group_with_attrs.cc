#include "task-spec/training_tensor_group_with_attrs.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("make_training_tensor_group_with_attrs_from_group_and_attrs") {
    TensorAttrs tensor_attrs = TensorAttrs{
      /*shape=*/TensorShape{
        /*dims=*/TensorDims{FFOrdered{
          8_p, 2_p, 3_p,
        }},
        /*data_type=*/DataType::FLOAT,
      },
      /*create_grad=*/CreateGrad::YES,
    };

    forward_tensor_guid_t forward_tensor = forward_tensor_guid_t{3};
    gradient_tensor_guid_t gradient_tensor = gradient_tensor_guid_t{5};
    std::vector<optimizer_tensor_guid_t> optimizer_tensors = {
      optimizer_tensor_guid_t{8},
      optimizer_tensor_guid_t{3},
    };

    TrainingTensorGroup training_tensor_group = TrainingTensorGroup{
      /*forward_tensor=*/forward_tensor,
      /*gradient_tensor=*/gradient_tensor,
      /*optimizer_tensors=*/optimizer_tensors,
    };

    TrainingTensorGroupWithAttrs result = make_training_tensor_group_with_attrs_from_group_and_attrs(training_tensor_group, tensor_attrs);
    TrainingTensorGroupWithAttrs correct = TrainingTensorGroupWithAttrs{
      /*tensor_attrs=*/tensor_attrs,
      /*forward_tensor=*/forward_tensor,
      /*gradient_tensor=*/gradient_tensor,
      /*optimizer_tensors=*/optimizer_tensors,
    };

    CHECK(result == correct);
  }

  TEST_CASE("tensor_group_without_attrs") {
    TensorAttrs tensor_attrs = TensorAttrs{
      /*shape=*/TensorShape{
        /*dims=*/TensorDims{FFOrdered{
          8_p, 2_p, 3_p,
        }},
        /*data_type=*/DataType::FLOAT,
      },
      /*create_grad=*/CreateGrad::YES,
    };

    forward_tensor_guid_t forward_tensor = forward_tensor_guid_t{3};
    gradient_tensor_guid_t gradient_tensor = gradient_tensor_guid_t{5};
    std::vector<optimizer_tensor_guid_t> optimizer_tensors = {
      optimizer_tensor_guid_t{8},
      optimizer_tensor_guid_t{3},
    };

    TrainingTensorGroupWithAttrs tensor_group_with_attrs = TrainingTensorGroupWithAttrs{
      /*tensor_attrs=*/tensor_attrs,
      /*forward_tensor=*/forward_tensor,
      /*gradient_tensor=*/gradient_tensor,
      /*optimizer_tensors=*/optimizer_tensors,
    };

    TrainingTensorGroup result = tensor_group_without_attrs(tensor_group_with_attrs);
    TrainingTensorGroup correct = TrainingTensorGroup{
      /*forward_tensor=*/forward_tensor,
      /*gradient_tensor=*/gradient_tensor,
      /*optimizer_tensors=*/optimizer_tensors,
    };

    CHECK(result == correct);
  }
}
