#ifndef _FLEXFLOW_KERNELS_TEST_SRC_INTERNAL_TEST_UTILS_H
#define _FLEXFLOW_KERNELS_TEST_SRC_INTERNAL_TEST_UTILS_H

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

GenericTensorAccessorW
    create_1d_accessor_w_with_contents(std::vector<float> const &contents,
                                       Allocator &allocator);
GenericTensorAccessorR
    create_1d_accessor_r_with_contents(std::vector<float> const &contents,
                                       Allocator &allocator);

GenericTensorAccessorW create_2d_accessor_w_with_contents(
    std::vector<std::vector<float>> const &contents, Allocator &allocator);
GenericTensorAccessorR create_2d_accessor_r_with_contents(
    std::vector<std::vector<float>> const &contents, Allocator &allocator);

GenericTensorAccessorW create_3d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<float>>> const &contents,
    Allocator &allocator);
GenericTensorAccessorR create_3d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<float>>> const &contents,
    Allocator &allocator);

GenericTensorAccessorW create_4d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &contents,
    Allocator &allocator);
GenericTensorAccessorR create_4d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &contents,
    Allocator &allocator);

bool contains_non_zero(GenericTensorAccessorR const &accessor);

void fill_with_zeros(GenericTensorAccessorW const &accessor);

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

} // namespace FlexFlow

#endif
