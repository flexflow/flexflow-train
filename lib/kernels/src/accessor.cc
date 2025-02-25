#include "kernels/accessor.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include <cstring>
#include <iostream>

namespace FlexFlow {

template <DataType DT>
struct AccessorDataIsEqual {
  bool operator()(GenericTensorAccessorR const &a,
                  GenericTensorAccessorR const &b) {
    int const num_elements = a.shape.num_elements().unwrap_nonnegative();
    if (num_elements != b.shape.num_elements().unwrap_nonnegative()) {
      return false;
    }

    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    auto cpu_a = copy_accessor_r_to_cpu_if_necessary(a, cpu_allocator);
    auto cpu_b = copy_accessor_r_to_cpu_if_necessary(b, cpu_allocator);

    using T = real_type_t<DT>;
    T const *a_ptr = cpu_a.get<DT>();
    T const *b_ptr = cpu_b.get<DT>();

    return std::equal(a_ptr, a_ptr + num_elements, b_ptr);
  }
};

bool accessor_data_is_equal(GenericTensorAccessorR const &accessor_a,
                            GenericTensorAccessorR const &accessor_b) {
  return DataTypeDispatch1<AccessorDataIsEqual>{}(
      accessor_a.data_type, accessor_a, accessor_b);
}

bool accessors_are_equal(GenericTensorAccessorR const &accessor_a,
                         GenericTensorAccessorR const &accessor_b) {
  return accessor_a.data_type == accessor_b.data_type &&
         accessor_a.device_type == accessor_b.device_type &&
         accessor_a.shape == accessor_b.shape &&
         accessor_data_is_equal(accessor_a, accessor_b);
}

GenericTensorAccessorW::operator GenericTensorAccessorR() const {
  return read_only_accessor_from_write_accessor(*this);
}

GenericTensorAccessorW::GenericTensorAccessorW(
    DataType data_type,
    ArrayShape const &shape,
    void *ptr,
    DeviceType device_type = DeviceType::GPU)
    : data_type(data_type), shape(shape), ptr(ptr), device_type(device_type) {}

std::tuple<DataType const &,
           ArrayShape const &,
           void *const &,
           DeviceType const &>
    GenericTensorAccessorW::tie() const {
  return std::tie(this->data_type, this->shape, this->ptr, this->device_type);
}

bool GenericTensorAccessorW::operator==(
    GenericTensorAccessorW const &other) const {
  return accessors_are_equal(*this, other);
}

bool GenericTensorAccessorW::operator!=(
    GenericTensorAccessorW const &other) const {
  return !(accessors_are_equal(*this, other));
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
  return fmt::format("<GenericTensorAccessorW data_type={} shape={} ptr={}>",
                     a.data_type,
                     a.shape,
                     a.ptr);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorW const &a) {
  return (s << fmt::to_string(a));
}

GenericTensorAccessorR::GenericTensorAccessorR(
    DataType data_type,
    ArrayShape const &shape,
    void const *ptr,
    DeviceType device_type = DeviceType::GPU)
    : data_type(data_type), shape(shape), ptr(ptr), device_type(device_type) {}

std::tuple<DataType const &,
           ArrayShape const &,
           void const *const &,
           DeviceType const &>
    GenericTensorAccessorR::tie() const {
  return std::tie(this->data_type, this->shape, this->ptr, this->device_type);
}

bool GenericTensorAccessorR::operator==(
    GenericTensorAccessorR const &other) const {
  return accessors_are_equal(*this, other);
}

bool GenericTensorAccessorR::operator!=(
    GenericTensorAccessorR const &other) const {
  return !(accessors_are_equal(*this, other));
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
  return fmt::format("<GenericTensorAccessorR data_type={} shape={} ptr={}>",
                     a.data_type,
                     a.shape,
                     a.ptr);
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

GenericTensorAccessorR read_only_accessor_from_write_accessor(
    GenericTensorAccessorW const &writable) {
  return GenericTensorAccessorR{writable.data_type,
                                writable.shape,
                                req<void const *>(writable.ptr),
                                writable.device_type};
}

bool is_shape_and_dtype_equal(GenericTensorAccessorR const &acc1,
                              GenericTensorAccessorR const &acc2) {
  return acc1.shape == acc2.shape && acc1.data_type == acc2.data_type;
}

bool shape_and_dtype_matches(GenericTensorAccessorR const &accessor,
                             ArrayShape const &expected_shape,
                             DataType const &expected_dtype) {
  return accessor.shape == expected_shape &&
         accessor.data_type == expected_dtype;
}

std::pair<ArrayShape, DataType>
    get_shape_and_datatype(GenericTensorAccessorR const &accessor) {
  return std::make_pair(accessor.shape, accessor.data_type);
}

} // namespace FlexFlow
