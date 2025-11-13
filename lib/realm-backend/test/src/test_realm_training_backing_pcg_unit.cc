// Unit tests for RealmTrainingBackingPCG
// These tests focus on individual functions and can run without full Realm runtime

#include <doctest/doctest.h>
#include "realm-backend/realm_training_backing_pcg.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/datatype.dtg.h"
#include "pcg/gpu_id_t.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "utils/integer_types.h"
#include <vector>
#include <cmath>

using namespace FlexFlow;

// Test utilities for tensor operations
TEST_SUITE("RealmTrainingBackingPCG - Tensor Operations") {
  
  TEST_CASE("calculate_tensor_size - FLOAT32") {
    TensorShape shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{10_n, 20_n}},
      DataType::FLOAT32
    };
    
    size_t expected_size = 10 * 20 * sizeof(float);
    size_t actual_size = calculate_tensor_size(shape, DataType::FLOAT32);
    
    CHECK_EQ(actual_size, expected_size);
  }
  
  TEST_CASE("calculate_tensor_size - FLOAT64") {
    TensorShape shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{5_n, 10_n}},
      DataType::FLOAT64
    };
    
    size_t expected_size = 5 * 10 * sizeof(double);
    size_t actual_size = calculate_tensor_size(shape, DataType::FLOAT64);
    
    CHECK_EQ(actual_size, expected_size);
  }
  
  TEST_CASE("calculate_tensor_size - INT32") {
    TensorShape shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{3_n, 4_n, 5_n}},
      DataType::INT32
    };
    
    size_t expected_size = 3 * 4 * 5 * sizeof(int32_t);
    size_t actual_size = calculate_tensor_size(shape, DataType::INT32);
    
    CHECK_EQ(actual_size, expected_size);
  }
  
  TEST_CASE("get_element_size - All Data Types") {
    CHECK_EQ(get_element_size(DataType::FLOAT32), sizeof(float));
    CHECK_EQ(get_element_size(DataType::FLOAT64), sizeof(double));
    CHECK_EQ(get_element_size(DataType::INT32), sizeof(int32_t));
    CHECK_EQ(get_element_size(DataType::INT64), sizeof(int64_t));
    CHECK_EQ(get_element_size(DataType::BOOL), sizeof(bool));
    CHECK_EQ(get_element_size(DataType::INT8), sizeof(int8_t));
    CHECK_EQ(get_element_size(DataType::UINT8), sizeof(uint8_t));
  }
}

TEST_SUITE("RealmTrainingBackingPCG - Data Parallel Distribution") {
  
  TEST_CASE("distribute_batch_data_parallel - Even Distribution") {
    TensorShape original_shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{100_n, 32_n}},
      DataType::FLOAT32
    };
    
    size_t num_devices = 4;
    std::vector<TensorShape> distributed = distribute_batch_data_parallel(original_shape, num_devices);
    
    CHECK_EQ(distributed.size(), num_devices);
    
    // Each device should get 25 samples
    for (size_t i = 0; i < num_devices; i++) {
      CHECK_EQ(distributed[i].dims.dims[0].size, 25_n);
      CHECK_EQ(distributed[i].dims.dims[1].size, 32_n);
      CHECK_EQ(distributed[i].data_type, DataType::FLOAT32);
    }
  }
  
  TEST_CASE("distribute_batch_data_parallel - Uneven Distribution") {
    TensorShape original_shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{100_n, 32_n}},
      DataType::FLOAT32
    };
    
    size_t num_devices = 3;
    std::vector<TensorShape> distributed = distribute_batch_data_parallel(original_shape, num_devices);
    
    CHECK_EQ(distributed.size(), num_devices);
    
    // First two devices get 33, last gets 34
    CHECK_EQ(distributed[0].dims.dims[0].size, 33_n);
    CHECK_EQ(distributed[1].dims.dims[0].size, 33_n);
    CHECK_EQ(distributed[2].dims.dims[0].size, 34_n);
  }
  
  TEST_CASE("distribute_batch_data_parallel - Batch Smaller Than Devices") {
    TensorShape original_shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{2_n, 32_n}},
      DataType::FLOAT32
    };
    
    size_t num_devices = 4;
    std::vector<TensorShape> distributed = distribute_batch_data_parallel(original_shape, num_devices);
    
    // Should return original shape when batch_per_device == 0
    CHECK_EQ(distributed.size(), 1u);
    CHECK_EQ(distributed[0].dims.dims[0].size, 2_n);
  }
  
  TEST_CASE("distribute_batch_data_parallel - Single Device") {
    TensorShape original_shape = TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{100_n, 32_n}},
      DataType::FLOAT32
    };
    
    size_t num_devices = 1;
    std::vector<TensorShape> distributed = distribute_batch_data_parallel(original_shape, num_devices);
    
    CHECK_EQ(distributed.size(), 1u);
    CHECK_EQ(distributed[0].dims.dims[0].size, 100_n);
  }
}

TEST_SUITE("RealmTrainingBackingPCG - Result Combination") {
  
  TEST_CASE("combine_parallel_results - Empty Results") {
    std::vector<float> empty_results;
    float result = combine_parallel_results(empty_results);
    CHECK_EQ(result, 0.0f);
  }
  
  TEST_CASE("combine_parallel_results - Single Result") {
    std::vector<float> results = {42.0f};
    float result = combine_parallel_results(results);
    CHECK_EQ(result, 42.0f);
  }
  
  TEST_CASE("combine_parallel_results - Multiple Results") {
    std::vector<float> results = {10.0f, 20.0f, 30.0f, 40.0f};
    float result = combine_parallel_results(results);
    float expected = (10.0f + 20.0f + 30.0f + 40.0f) / 4.0f;
    CHECK_EQ(result, expected);
  }
  
  TEST_CASE("combine_parallel_results - Negative Values") {
    std::vector<float> results = {-10.0f, 20.0f, -30.0f};
    float result = combine_parallel_results(results);
    float expected = (-10.0f + 20.0f - 30.0f) / 3.0f;
    CHECK_EQ(result, expected);
  }
  
  TEST_CASE("combine_parallel_results - Zero Values") {
    std::vector<float> results = {0.0f, 0.0f, 0.0f};
    float result = combine_parallel_results(results);
    CHECK_EQ(result, 0.0f);
  }
}

TEST_SUITE("RealmTrainingBackingPCG - Conversion Functions") {
  
  TEST_CASE("convert_parallel_to_regular_layer") {
    parallel_layer_guid_t parallel_layer{DataflowNode{0}};
    layer_guid_t regular_layer = convert_parallel_to_regular_layer(parallel_layer);
    
    CHECK_EQ(regular_layer.raw_node.raw_value, 0u);
  }
  
  TEST_CASE("convert_parallel_to_regular_tensor") {
    parallel_tensor_guid_t parallel_tensor{DataflowOutput{DataflowNode{0}, 0_n}};
    tensor_guid_t regular_tensor = convert_parallel_to_regular_tensor(parallel_tensor);
    
    CHECK_EQ(regular_tensor.raw_graph_output.node.raw_value, 0u);
    CHECK_EQ(regular_tensor.raw_graph_output.output_idx, 0_n);
  }
}

TEST_SUITE("RealmTrainingBackingPCG - Floating Point Comparison") {
  
  TEST_CASE("float_equal_with_tolerance - Equal Values") {
    CHECK(float_equal_with_tolerance(1.0f, 1.0f));
    CHECK(float_equal_with_tolerance(0.0f, 0.0f));
    CHECK(float_equal_with_tolerance(-1.0f, -1.0f));
  }
  
  TEST_CASE("float_equal_with_tolerance - Different Values") {
    CHECK_FALSE(float_equal_with_tolerance(1.0f, 2.0f));
    CHECK_FALSE(float_equal_with_tolerance(0.0f, 0.1f));
  }
  
  TEST_CASE("double_equal_with_tolerance - Equal Values") {
    CHECK(double_equal_with_tolerance(1.0, 1.0));
    CHECK(double_equal_with_tolerance(0.0, 0.0));
  }
  
  TEST_CASE("combine_float_values_with_tolerance - Equal Values") {
    float result = combine_float_values_with_tolerance(1.0f, 1.0f);
    CHECK_EQ(result, 1.0f);
  }
  
  TEST_CASE("combine_float_values_with_tolerance - Different Values Throws") {
    CHECK_THROWS_AS(
      combine_float_values_with_tolerance(1.0f, 2.0f),
      std::runtime_error
    );
  }
}

