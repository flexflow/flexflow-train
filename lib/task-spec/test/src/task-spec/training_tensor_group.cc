#include <doctest/doctest.h>
#include "task-spec/training_tensor_group.h"
#include "test/utils/doctest/fmt/unordered_set.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_all_training_tensors_in_tensor_group") {
    forward_tensor_guid_t forward_tensor = forward_tensor_guid_t{3};
    gradient_tensor_guid_t gradient_tensor = gradient_tensor_guid_t{5};
    optimizer_tensor_guid_t optimizer_tensor1 = optimizer_tensor_guid_t{8};
    optimizer_tensor_guid_t optimizer_tensor2 = optimizer_tensor_guid_t{3};

    std::vector<optimizer_tensor_guid_t> optimizer_tensors = {
      optimizer_tensor1,
      optimizer_tensor2,
    };

    TrainingTensorGroup training_tensor_group = TrainingTensorGroup{
      /*forward_tensor=*/forward_tensor,
      /*gradient_tensor=*/gradient_tensor,
      /*optimizer_tensors=*/optimizer_tensors,
    };

    std::unordered_set<training_tensor_guid_t> result = get_all_training_tensors_in_tensor_group(training_tensor_group);
    std::unordered_set<training_tensor_guid_t> correct = {
      training_tensor_guid_t{forward_tensor},
      training_tensor_guid_t{gradient_tensor},
      training_tensor_guid_t{optimizer_tensor1},
      training_tensor_guid_t{optimizer_tensor2},
    };

    CHECK(result == correct);
  }
}
