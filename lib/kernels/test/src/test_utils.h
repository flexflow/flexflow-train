#ifndef _FLEXFLOW_KERNELS_TEST_UTILS
#define _FLEXFLOW_KERNELS_TEST_UTILS

#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/device.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/datatype.h"
#include "op-attrs/datatype_value.dtg.h"
#include <doctest/doctest.h>
#include <sstream>
#include <string>
#include <vector>

namespace FlexFlow {

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator);

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator);

GenericTensorAccessorR create_zero_filled_accessor_r(TensorShape const &shape,
                                                     Allocator &allocator);

TensorShape make_tensor_shape_from_legion_dims(FFOrdered<nonnegative_int> dims,
                                               DataType DT);

bool contains_non_zero(GenericTensorAccessorR const &accessor);

void fill_with_zeros(GenericTensorAccessorW const &accessor);

GenericTensorAccessorW
    copy_accessor_w_to_cpu_if_necessary(GenericTensorAccessorW const &accessor,
                                        Allocator &allocator);

GenericTensorAccessorR
    copy_accessor_r_to_cpu_if_necessary(GenericTensorAccessorR const &accessor,
                                        Allocator &allocator);

void print_2d_tensor_accessor_contents(GenericTensorAccessorR const &accessor,
                                       std::ostream &stream);

bool accessors_are_equal(GenericTensorAccessorR const &accessor_a,
                         GenericTensorAccessorR const &accessor_b);

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val);

GenericTensorAccessorR create_filled_accessor_r(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val);

template <typename T, typename Func>
std::vector<T> repeat(std::size_t n, Func &&func) {
  std::vector<T> result;
  // result.reserve(n); // Sometimes we don't have default constructor for T
  for (std::size_t i = 0; i < n; ++i) {
    result.push_back(func());
  }
  return result;
}

} // namespace FlexFlow

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
