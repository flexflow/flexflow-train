// Enhanced test to verify true parallelism in PCG wrapper
#include "realm-backend/realm_training_backing_pcg.h"
#include "realm-backend/realm_training_backing.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>

namespace FlexFlow {
namespace Testing {

// Test to verify true parallel execution
class ParallelismVerificationTest {
private:
  std::atomic<int> concurrent_executions{0};
  std::atomic<int> max_concurrent_executions{0};
  std::chrono::steady_clock::time_point start_time;
  
public:
  // Test 1: Verify concurrent task execution
  void test_concurrent_execution() {
    std::cout << "=== Testing Concurrent Execution ===" << std::endl;
    
    // Create PCG backing with multiple devices
    RealmTrainingBackingPCG backing = create_multi_device_backing();
    
    // Get a test layer
    parallel_layer_guid_t test_layer = get_test_layer_for_parallelism();
    
    // Hook into task execution to monitor concurrency
    setup_concurrency_monitoring();
    
    // Execute forward pass
    start_time = std::chrono::steady_clock::now();
    Future<float> result = execute_forward_pcg(backing, test_layer);
    
    // Wait for completion
    float value = result.get();
    
    // Check results
    std::cout << "Max concurrent executions: " << max_concurrent_executions.load() << std::endl;
    std::cout << "Result: " << value << std::endl;
    
    // Verify true parallelism
    if (max_concurrent_executions.load() > 1) {
      std::cout << "✅ TRUE PARALLELISM DETECTED!" << std::endl;
    } else {
      std::cout << "❌ No parallelism detected - tasks executed sequentially" << std::endl;
    }
  }
  
  // Test 2: Compare execution times
  void test_execution_time_comparison() {
    std::cout << "=== Testing Execution Time Comparison ===" << std::endl;
    
    // Create single-device backing
    RealmTrainingBackingPCG single_device_backing = create_single_device_backing();
    
    // Create multi-device backing
    RealmTrainingBackingPCG multi_device_backing = create_multi_device_backing();
    
    parallel_layer_guid_t test_layer = get_test_layer_for_parallelism();
    
    // Time single-device execution
    auto single_start = std::chrono::high_resolution_clock::now();
    Future<float> single_result = execute_forward_pcg(single_device_backing, test_layer);
    float single_value = single_result.get();
    auto single_end = std::chrono::high_resolution_clock::now();
    
    // Time multi-device execution
    auto multi_start = std::chrono::high_resolution_clock::now();
    Future<float> multi_result = execute_forward_pcg(multi_device_backing, test_layer);
    float multi_value = multi_result.get();
    auto multi_end = std::chrono::high_resolution_clock::now();
    
    // Calculate execution times
    auto single_duration = std::chrono::duration_cast<std::chrono::microseconds>(single_end - single_start);
    auto multi_duration = std::chrono::duration_cast<std::chrono::microseconds>(multi_end - multi_start);
    
    std::cout << "Single device time: " << single_duration.count() << " microseconds" << std::endl;
    std::cout << "Multi device time: " << multi_duration.count() << " microseconds" << std::endl;
    
    // Calculate speedup
    double speedup = static_cast<double>(single_duration.count()) / multi_duration.count();
    std::cout << "Speedup: " << speedup << "x" << std::endl;
    
    // Verify speedup (should be > 1.0 for true parallelism)
    if (speedup > 1.1) {  // Allow for some overhead
      std::cout << "✅ PARALLEL SPEEDUP ACHIEVED!" << std::endl;
    } else {
      std::cout << "❌ No speedup detected - may not be truly parallel" << std::endl;
    }
  }
  
  // Test 3: Device utilization verification
  void test_device_utilization() {
    std::cout << "=== Testing Device Utilization ===" << std::endl;
    
    RealmTrainingBackingPCG backing = create_multi_device_backing();
    parallel_layer_guid_t test_layer = get_test_layer_for_parallelism();
    
    // Get devices assigned to this layer
    std::vector<device_id_t> devices = get_layer_devices(backing, test_layer);
    
    std::cout << "Layer assigned to " << devices.size() << " devices:" << std::endl;
    for (device_id_t device : devices) {
      std::cout << "  - Device " << device.gpu_id.gpu_index.raw_value << std::endl;
    }
    
    // Execute and monitor device activity
    std::vector<std::atomic<bool>> device_active(devices.size());
    for (auto& active : device_active) {
      active.store(false);
    }
    
    // Hook into device processors to monitor activity
    setup_device_monitoring(devices, device_active);
    
    // Execute forward pass
    Future<float> result = execute_forward_pcg(backing, test_layer);
    float value = result.get();
    
    // Check device utilization
    int active_devices = 0;
    for (size_t i = 0; i < device_active.size(); i++) {
      if (device_active[i].load()) {
        active_devices++;
        std::cout << "  ✅ Device " << i << " was active" << std::endl;
      } else {
        std::cout << "  ❌ Device " << i << " was NOT active" << std::endl;
      }
    }
    
    std::cout << "Active devices: " << active_devices << "/" << devices.size() << std::endl;
    
    if (active_devices > 1) {
      std::cout << "✅ MULTI-DEVICE UTILIZATION CONFIRMED!" << std::endl;
    } else {
      std::cout << "❌ Only single device utilized" << std::endl;
    }
  }
  
  // Test 4: Result consistency verification
  void test_result_consistency() {
    std::cout << "=== Testing Result Consistency ===" << std::endl;
    
    // Create non-PCG backing for reference
    RealmTrainingBacking reference_backing = create_reference_backing();
    
    // Create PCG backing
    RealmTrainingBackingPCG pcg_backing = create_multi_device_backing();
    
    // Execute same computation on both
    layer_guid_t reference_layer = get_reference_layer();
    parallel_layer_guid_t pcg_layer = get_corresponding_pcg_layer(reference_layer);
    
    Future<float> reference_result = execute_forward(reference_backing, reference_layer);
    Future<float> pcg_result = execute_forward_pcg(pcg_backing, pcg_layer);
    
    float reference_value = reference_result.get();
    float pcg_value = pcg_result.get();
    
    std::cout << "Reference result: " << reference_value << std::endl;
    std::cout << "PCG result: " << pcg_value << std::endl;
    
    float difference = std::abs(reference_value - pcg_value);
    std::cout << "Difference: " << difference << std::endl;
    
    if (difference < 1e-5) {
      std::cout << "✅ RESULTS CONSISTENT!" << std::endl;
    } else {
      std::cout << "❌ Results differ - potential correctness issue" << std::endl;
    }
  }
  
  // Test 5: Parallel combination verification
  void test_parallel_combination() {
    std::cout << "=== Testing Parallel Result Combination ===" << std::endl;
    
    // Create multiple mock device futures
    std::vector<Future<float>> device_futures;
    
    // Create promises for different devices
    std::vector<std::unique_ptr<Promise<float>>> promises;
    for (int i = 0; i < 3; i++) {
      auto promise = std::make_unique<Promise<float>>();
      device_futures.push_back(promise->get_future());
      promises.push_back(std::move(promise));
    }
    
    // Set different values on each device
    std::vector<float> device_values = {1.0f, 2.0f, 3.0f};
    for (size_t i = 0; i < promises.size(); i++) {
      promises[i]->set_value(device_values[i]);
    }
    
    // Test combination
    Future<float> combined_result = combine_device_results_parallel(device_futures);
    float combined_value = combined_result.get();
    
    // Expected result for data parallelism: average = (1+2+3)/3 = 2.0
    float expected_value = 2.0f;
    
    std::cout << "Combined result: " << combined_value << std::endl;
    std::cout << "Expected result: " << expected_value << std::endl;
    
    if (std::abs(combined_value - expected_value) < 1e-5) {
      std::cout << "✅ PARALLEL COMBINATION WORKS!" << std::endl;
    } else {
      std::cout << "❌ Parallel combination failed" << std::endl;
    }
  }
  
  // Run all tests
  void run_all_tests() {
    std::cout << "Starting Parallelism Verification Tests..." << std::endl;
    
    try {
      test_concurrent_execution();
      test_execution_time_comparison();
      test_device_utilization();
      test_result_consistency();
      test_parallel_combination();
      
      std::cout << "All parallelism tests completed!" << std::endl;
    } catch (const std::exception& e) {
      std::cout << "Test failed with exception: " << e.what() << std::endl;
    }
  }
  
private:
  // Helper implementations would go here
  void setup_concurrency_monitoring() {
    // Hook into task execution to monitor concurrent executions
    // This would need to be implemented based on your task execution infrastructure
  }
  
  void setup_device_monitoring(std::vector<device_id_t> const& devices, 
                              std::vector<std::atomic<bool>>& device_active) {
    // Hook into device processors to monitor activity
    // This would need to be implemented based on your device infrastructure
  }
  
  RealmTrainingBackingPCG create_multi_device_backing() {
    // Create backing with multiple devices
    // This would need to be implemented based on your creation utilities
    return RealmTrainingBackingPCG{};
  }
  
  RealmTrainingBackingPCG create_single_device_backing() {
    // Create backing with single device
    // This would need to be implemented based on your creation utilities
    return RealmTrainingBackingPCG{};
  }
  
  RealmTrainingBacking create_reference_backing() {
    // Create non-PCG backing for reference
    // This would need to be implemented based on your creation utilities
    return RealmTrainingBacking{};
  }
  
  parallel_layer_guid_t get_test_layer_for_parallelism() {
    // Get a layer that can be parallelized
    // This would need to be implemented based on your layer creation utilities
    return parallel_layer_guid_t{};
  }
  
  layer_guid_t get_reference_layer() {
    // Get reference layer for comparison
    // This would need to be implemented based on your layer creation utilities
    return layer_guid_t{};
  }
  
  parallel_layer_guid_t get_corresponding_pcg_layer(layer_guid_t const& layer) {
    // Convert reference layer to PCG layer
    return convert_regular_to_parallel_layer(layer);
  }
};

} // namespace Testing
} // namespace FlexFlow

// Main test runner
int main() {
  FlexFlow::Testing::ParallelismVerificationTest test;
  test.run_all_tests();
  return 0;
} 