#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/ff_ordered/get_idxs.h"
#include "op-attrs/tensor_dims_coord.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/reversed.h"
#include "utils/containers/vector_of.h"
#include "utils/hash/tuple.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

nonnegative_int calculate_accessor_offset(TensorDimsCoord const &coord,
                                          TensorDims const &tensor_dims) {
  ASSERT(tensor_dims_coord_get_num_dims(coord) == get_num_dims(tensor_dims),
         "Number of indices does not match the number of dimensions");

  nonnegative_int offset = 0_n;
  positive_int multiplier = 1_p;

  for (ff_dim_t dim : reversed(vector_of(get_idxs(tensor_dims.ff_ordered)))) {
    ASSERT(coord.ff_ordered.at(dim) < dim_at_idx(tensor_dims, dim),
           "Out of bounds access",
           dim);

    offset += coord.ff_ordered.at(dim) * multiplier;
    multiplier *= tensor_dims.ff_ordered.at(dim);
  }

  return offset;
}

TensorShape
    get_tensor_shape_for_accessor_r(GenericTensorAccessorR const &accessor) {
  return accessor.shape;
}

TensorShape
    get_tensor_shape_for_accessor_w(GenericTensorAccessorW const &accessor) {
  return accessor.shape;
}

void copy_accessor_data_to_l_from_r(
    GenericTensorAccessorW const &dst_accessor,
    GenericTensorAccessorR const &src_accessor) {
  size_t num_bytes = get_size_in_bytes(dst_accessor.shape)
                         .unwrap_num_bytes()
                         .unwrap_nonnegative();

  DeviceType dst_device_type = dst_accessor.device_type;
  DeviceType src_device_type = src_accessor.device_type;

  if (src_device_type == DeviceType::CPU &&
      dst_device_type == DeviceType::CPU) {
    memcpy(dst_accessor.ptr, src_accessor.ptr, num_bytes);
  } else if (src_device_type == DeviceType::CPU &&
             dst_device_type == DeviceType::GPU) {
    checkCUDA(cudaMemcpy(
        dst_accessor.ptr, src_accessor.ptr, num_bytes, cudaMemcpyHostToDevice));
  } else if (src_device_type == DeviceType::GPU &&
             dst_device_type == DeviceType::CPU) {
    checkCUDA(cudaMemcpy(
        dst_accessor.ptr, src_accessor.ptr, num_bytes, cudaMemcpyDeviceToHost));
  } else {
    assert(src_device_type == DeviceType::GPU);
    assert(dst_device_type == DeviceType::GPU);
    checkCUDA(cudaMemcpy(dst_accessor.ptr,
                         src_accessor.ptr,
                         num_bytes,
                         cudaMemcpyDeviceToDevice));
  }
}

GenericTensorAccessorW::operator GenericTensorAccessorR() const {
  return read_only_accessor_from_write_accessor(*this);
}

GenericTensorAccessorW::GenericTensorAccessorW(
    TensorShape const &shape,
    void *ptr,
    DeviceType device_type = DeviceType::GPU)
    : shape(shape), ptr(ptr), device_type(device_type) {}

std::tuple<TensorShape const &, void *const &, DeviceType const &>
    GenericTensorAccessorW::tie() const {
  return std::tie(this->shape, this->ptr, this->device_type);
}

bool GenericTensorAccessorW::operator==(
    GenericTensorAccessorW const &other) const {
  return this->tie() == other.tie();
}

bool GenericTensorAccessorW::operator!=(
    GenericTensorAccessorW const &other) const {
  return this->tie() != other.tie();
}

int32_t *GenericTensorAccessorW::get_int32_ptr() const {
  return this->get<DataType::INT32>();
}

int64_t *GenericTensorAccessorW::get_int64_ptr() const {
  return this->get<DataType::INT64>();
}

float *GenericTensorAccessorW::get_float_ptr() const {
  return this->get<DataType::FLOAT>();
}

double *GenericTensorAccessorW::get_double_ptr() const {
  return this->get<DataType::DOUBLE>();
}

half *GenericTensorAccessorW::get_half_ptr() const {
  return this->get<DataType::HALF>();
}

std::string format_as(GenericTensorAccessorW const &a) {
  return fmt::format("<GenericTensorAccessorW shape={} ptr={} device_type={}>",
                     a.shape,
                     a.ptr,
                     a.device_type);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorW const &a) {
  return (s << fmt::to_string(a));
}

GenericTensorAccessorR::GenericTensorAccessorR(
    TensorShape const &shape,
    void const *ptr,
    DeviceType device_type = DeviceType::GPU)
    : shape(shape), ptr(ptr), device_type(device_type) {}

std::tuple<TensorShape const &, void const *const &, DeviceType const &>
    GenericTensorAccessorR::tie() const {
  return std::tie(this->shape, this->ptr, this->device_type);
}

bool GenericTensorAccessorR::operator==(
    GenericTensorAccessorR const &other) const {
  return this->tie() == other.tie();
}

bool GenericTensorAccessorR::operator!=(
    GenericTensorAccessorR const &other) const {
  return this->tie() != other.tie();
}

int32_t const *GenericTensorAccessorR::get_int32_ptr() const {
  return this->get<DataType::INT32>();
}

int64_t const *GenericTensorAccessorR::get_int64_ptr() const {
  return this->get<DataType::INT64>();
}

float const *GenericTensorAccessorR::get_float_ptr() const {
  return this->get<DataType::FLOAT>();
}

double const *GenericTensorAccessorR::get_double_ptr() const {
  return this->get<DataType::DOUBLE>();
}

half const *GenericTensorAccessorR::get_half_ptr() const {
  return get<DataType::HALF>();
}

std::string format_as(GenericTensorAccessorR const &a) {
  return fmt::format("<GenericTensorAccessorR shape={} ptr={} device_type={}>",
                     a.shape,
                     a.ptr,
                     a.device_type);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorR const &a) {
  return (s << fmt::to_string(a));
}

int32_t const *get_int32_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::INT32>(a);
}

int64_t const *get_int64_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::INT64>(a);
}

float const *get_float_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::FLOAT>(a);
}

double const *get_double_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::DOUBLE>(a);
}

half const *get_half_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::HALF>(a);
}

std::vector<int32_t const *>
    get_int32_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::INT32>(a);
}

std::vector<int64_t const *>
    get_int64_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::INT64>(a);
}

std::vector<float const *>
    get_float_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::FLOAT>(a);
}

std::vector<double const *>
    get_double_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::DOUBLE>(a);
}

std::vector<half const *>
    get_half_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::HALF>(a);
}

int32_t *get_int32_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::INT32>(a);
}

int64_t *get_int64_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::INT64>(a);
}

float *get_float_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::FLOAT>(a);
}

double *get_double_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::DOUBLE>(a);
}

half *get_half_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::HALF>(a);
}

std::vector<int32_t *>
    get_int32_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::INT32>(a);
}

std::vector<int64_t *>
    get_int64_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::INT64>(a);
}

std::vector<float *>
    get_float_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::FLOAT>(a);
}

std::vector<double *>
    get_double_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::DOUBLE>(a);
}

std::vector<half *>
    get_half_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::HALF>(a);
}

GenericTensorAccessorR read_only_accessor_from_write_accessor(
    GenericTensorAccessorW const &writable) {
  return GenericTensorAccessorR{
      writable.shape,
      writable.ptr,
      writable.device_type,
  };
}

bool accessors_have_same_shape(GenericTensorAccessorR const &acc1,
                               GenericTensorAccessorR const &acc2) {
  return acc1.shape == acc2.shape;
}

bool accessors_have_same_shape(GenericTensorAccessorW const &acc1,
                               GenericTensorAccessorW const &acc2) {
  return acc1.shape == acc2.shape;
}

template int32_t
    accessor_get_only_value<DataType::INT32>(GenericTensorAccessorR const &);

} // namespace FlexFlow

namespace std {

using namespace ::FlexFlow;

size_t hash<GenericTensorAccessorR>::operator()(
    GenericTensorAccessorR const &a) const {
  return get_std_hash(a.tie());
}

size_t hash<GenericTensorAccessorW>::operator()(
    GenericTensorAccessorW const &a) const {
  return get_std_hash(a.tie());
}

} // namespace std
