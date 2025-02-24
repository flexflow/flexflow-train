#include "local-execution/local_cpu_allocator.h"
#include "local-execution/local_tensor_backing.h"
#include "test_utils.h"
#include "utils/containers/keys.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

bool is_shape_and_dtype_equal_for_tensor_backings(
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &m1,
    std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> const &m2) {
  if (keys(m1) == keys(m2)) {
    for (std::pair<TensorTypeVariant, GenericTensorAccessorW> const
             &tensor_type_backing : m1) {
      if (is_shape_and_dtype_equal(tensor_type_backing.second,
                                   m2.at(tensor_type_backing.first))) {
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
  TEST_CASE("LocalTensorBacking") {
    MockTensorGuidSource tensor_guid_source;
    GradientTensorSource gradient_tensor_source;
    OptimizerTensorSource optimizer_tensor_source;
    LossTensorSource loss_tensor_source;

    SUBCASE("merge_optimizer_mappings") {
      SUBCASE("Both empty") {
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            result = merge_optimizer_mappings({}, {});
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            correct = {};
        CHECK(result == correct);
      }

      tensor_guid_t allocated_tensor_guid =
          tensor_guid_source.new_mock_tensor_guid();
      optimizer_tensor_t optimizer_tensor_1 =
          optimizer_tensor_source.new_optimizer_tensor();
      optimizer_tensor_t optimizer_tensor_2 =
          optimizer_tensor_source.new_optimizer_tensor();
      std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
          correct = {{allocated_tensor_guid,
                      {optimizer_tensor_1, optimizer_tensor_2}}};
      SUBCASE("Unallocated is empty") {
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            allocated = {{allocated_tensor_guid,
                          {optimizer_tensor_1, optimizer_tensor_2}}};
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            result = merge_optimizer_mappings(allocated, {});
        CHECK(result == correct);
      }
      SUBCASE("Allocated is empty") {
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            unallocated = {{allocated_tensor_guid,
                            {optimizer_tensor_1, optimizer_tensor_2}}};
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            result = merge_optimizer_mappings({}, unallocated);
        CHECK(result == correct);
      }

      SUBCASE("Both are partially allocated") {
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            allocated = {{allocated_tensor_guid, {optimizer_tensor_1}}};
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            unallocated = {{allocated_tensor_guid, {optimizer_tensor_2}}};
        std::unordered_map<tensor_guid_t, std::vector<optimizer_tensor_t>>
            result = merge_optimizer_mappings(allocated, unallocated);
        CHECK(result == correct);
      }
    }

    SUBCASE("get_tensor_backings") {
      Allocator allocator = create_local_cpu_memory_allocator();
      SUBCASE("Both are empty") {
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> result =
            get_tensor_backings({}, {}, allocator);
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> correct =
            {};
        CHECK(result == correct);
      }

      tensor_guid_t allocated_tensor_guid =
          tensor_guid_source.new_mock_tensor_guid();
      tensor_guid_t unallocated_tensor_guid =
          tensor_guid_source.new_mock_tensor_guid();

      TensorAttrs allocated_tensor_attrs = TensorAttrs{
          TensorShape{TensorDims{FFOrdered<nonnegative_int>{16_n, 10_n}},
                      DataType::FLOAT},
          CreateGrad::NO};
      TensorAttrs unallocated_tensor_attrs = TensorAttrs{
          TensorShape{TensorDims{FFOrdered<nonnegative_int>{16_n, 20_n}},
                      DataType::FLOAT},
          CreateGrad::YES};

      GenericTensorAccessorW allocated_tensor_backing =
          allocator.allocate_tensor(allocated_tensor_attrs.shape);
      GenericTensorAccessorW unallocated_tensor_backing =
          allocator.allocate_tensor(unallocated_tensor_attrs.shape);

      SUBCASE("Unallocated is empty") {
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
            allocated = {{TensorTypeVariant{allocated_tensor_guid},
                          allocated_tensor_backing}};
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> result =
            get_tensor_backings(allocated, {}, allocator);
        CHECK(result == allocated);
      }
      SUBCASE("Allocated is empty") {
        std::unordered_map<TensorTypeVariant, TensorShape> unallocated = {
            {TensorTypeVariant{unallocated_tensor_guid},
             unallocated_tensor_attrs.shape}};
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> result =
            get_tensor_backings({}, unallocated, allocator);
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> correct =
            {{TensorTypeVariant{unallocated_tensor_guid},
              unallocated_tensor_backing}};
        CHECK(is_shape_and_dtype_equal_for_tensor_backings(result, correct));
      }
      SUBCASE("Both are partially allocated") {
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW>
            allocated = {{TensorTypeVariant{allocated_tensor_guid},
                          allocated_tensor_backing}};
        std::unordered_map<TensorTypeVariant, TensorShape> unallocated = {
            {TensorTypeVariant{unallocated_tensor_guid},
             unallocated_tensor_attrs.shape}};

        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> result =
            get_tensor_backings(allocated, unallocated, allocator);
        std::unordered_map<TensorTypeVariant, GenericTensorAccessorW> correct =
            {{TensorTypeVariant{allocated_tensor_guid},
              allocated_tensor_backing},
             {TensorTypeVariant{unallocated_tensor_guid},
              unallocated_tensor_backing}};
        CHECK(is_shape_and_dtype_equal_for_tensor_backings(result, correct));
      }
    }
  }
}
