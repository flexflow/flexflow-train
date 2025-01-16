#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/device.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include <doctest/doctest.h>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using namespace FlexFlow;

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator,
                                                       bool cpu_fill = false);

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                float val,
                                                bool cpu_fill = false);

GenericTensorAccessorW create_iota_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator,
                                                     bool cpu_fill = false);

void fill_tensor_accessor_w(GenericTensorAccessorW accessor,
                            float val,
                            bool cpu_fill = false);

TensorShape make_float_tensor_shape_from_legion_dims(FFOrdered<size_t> dims);

TensorShape make_double_tensor_shape_from_legion_dims(FFOrdered<size_t> dims);

template <typename T>
std::vector<T> load_data_to_host_from_device(GenericTensorAccessorR accessor) {
  int volume = accessor.shape.get_volume();

  std::vector<T> local_data(volume);
  checkCUDA(cudaMemcpy(local_data.data(),
                       accessor.ptr,
                       local_data.size() * sizeof(T),
                       cudaMemcpyDeviceToHost));
  return local_data;
}

template <typename T>
bool contains_non_zero(std::vector<T> &data) {
  return !all_of(
      data.begin(), data.end(), [](T const &val) { return val == 0; });
}

template <typename T, typename Func>
std::vector<T> repeat(std::size_t n, Func &&func) {
  std::vector<T> result;
  // result.reserve(n); // Sometimes we don't have default constructor for T
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(func());
  }
  return result;
}

// Specialize doctest's StringMaker for std::vector<float>
template <>
struct doctest::StringMaker<std::vector<float>> {
  static doctest::String convert(std::vector<float> const &vec) {
    std::ostringstream oss;
    for (size_t i = 0; i < vec.size(); ++i) {
      oss << vec[i];
      if (i != vec.size() - 1) {
        oss << ", ";
      }
    }
    return doctest::String(("[" + oss.str() + "]").c_str());
  }
};

#endif
