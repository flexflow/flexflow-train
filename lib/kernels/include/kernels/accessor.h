#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSOR_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_ACCESSOR_H

#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "kernels/legion_dim.h"
#include "kernels/legion_ordered/legion_ordered.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_dims.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "pcg/device_type.dtg.h"
#include "utils/containers/transform.h"
#include <libassert/assert.hpp>
#include <string>
#include "op-attrs/tensor_dims.h"

namespace FlexFlow {

nonnegative_int
    calculate_accessor_offset(TensorDimsCoord const &,
                              TensorDims const &);

class GenericTensorAccessorR {
public:
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type const *get() const {
    ASSERT(this->shape.data_type == DT, "Invalid datatype requested");

    return static_cast<real_type_t<DT> const *>(this->ptr);
  }

  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  half const *get_half_ptr() const;

  GenericTensorAccessorR() = delete;

  GenericTensorAccessorR(TensorShape const &shape,
                         void const *ptr,
                         DeviceType device_type);

  bool operator==(GenericTensorAccessorR const &) const;
  bool operator!=(GenericTensorAccessorR const &) const;

  template <DataType DT>
  real_type_t<DT> const &
      at(TensorDimsCoord const &indices) const {
    ASSERT(this->device_type == DeviceType::CPU,
           "GenericTensorAccessorR::at() requires CPU-allocated tensor");
    ASSERT(this->shape.data_type == DT, "Invalid datatype requested");

    using T = real_type_t<DT>;
    T const *data_ptr = static_cast<T const *>(this->ptr);
    nonnegative_int offset = calculate_accessor_offset(indices, this->shape.dims);
    return data_ptr[offset.unwrap_nonnegative()];
  }

public:
  TensorShape shape;
  void const *ptr;
  DeviceType device_type;

private:
  std::tuple<decltype(shape) const &,
             decltype(ptr) const &,
             decltype(device_type) const &>
      tie() const;
};

std::string format_as(GenericTensorAccessorR const &);
std::ostream &operator<<(std::ostream &, GenericTensorAccessorR const &);

class GenericTensorAccessorW {
public:
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type *get() const {
    ASSERT(this->shape.data_type == DT, "Invalid datatype requested");

    return static_cast<real_type_t<DT> *>(this->ptr);
  }

  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;

  GenericTensorAccessorW() = delete;

  GenericTensorAccessorW(TensorShape const &shape,
                         void *ptr,
                         DeviceType device_type);

  bool operator==(GenericTensorAccessorW const &) const;
  bool operator!=(GenericTensorAccessorW const &) const;

  operator GenericTensorAccessorR() const;

  template <DataType DT>
  real_type_t<DT> &at(TensorDimsCoord const &indices) {
    ASSERT(this->device_type == DeviceType::CPU,
           "GenericTensorAccessorW::at() requires CPU-allocated tensor");
    ASSERT(this->shape.data_type == DT, "Invalid datatype requested");

    using T = real_type_t<DT>;
    T *data_ptr = static_cast<T *>(this->ptr);
    nonnegative_int offset = calculate_accessor_offset(indices, this->shape.dims);
    return data_ptr[offset.unwrap_nonnegative()];
  }

  template <DataType DT>
  real_type_t<DT> &at(TensorDimsCoord const &indices) const {
    ASSERT(this->device_type == DeviceType::CPU,
           "GenericTensorAccessorW::at() requires CPU-allocated tensor");
    ASSERT(this->shape.data_type == DT, "Invalid datatype requested");

    using T = real_type_t<DT>;
    T *data_ptr = static_cast<T *>(this->ptr);
    nonnegative_int offset = calculate_accessor_offset(indices, this->shape.dims);
    return data_ptr[offset.unwrap_nonnegative()];
  }

public:
  TensorShape shape;
  void *ptr;
  DeviceType device_type;

private:
  std::tuple<decltype(shape) const &,
             decltype(ptr) const &,
             decltype(device_type) const &>
      tie() const;
};

std::string format_as(GenericTensorAccessorW const &);
std::ostream &operator<<(std::ostream &, GenericTensorAccessorW const &);

template <DataType DT>
typename data_type_enum_to_class<DT>::type *
    get(GenericTensorAccessorW const &a) {
  ASSERT(a.shape.data_type == DT, "Invalid datatype requested");
  return static_cast<real_type_t<DT> *>(a.ptr);
}

template <DataType DT>
std::vector<real_type_t<DT> *>
    get(std::vector<GenericTensorAccessorW> const &accs) {
  std::vector<real_type_t<DT> *> out;
  for (auto acc : accs) {
    out.push_back(get<DT>(acc));
  }
  return out;
}

template <DataType DT>
typename data_type_enum_to_class<DT>::type const *
    get(GenericTensorAccessorR const &a) {
  ASSERT(a.shape.data_type == DT, "Invalid datatype requested");
  return static_cast<real_type_t<DT> const *>(a.ptr);
}

int32_t const *get_int32_ptr(GenericTensorAccessorR const &);
int64_t const *get_int64_ptr(GenericTensorAccessorR const &);
float const *get_float_ptr(GenericTensorAccessorR const &);
double const *get_double_ptr(GenericTensorAccessorR const &);
half const *get_half_ptr(GenericTensorAccessorR const &);
std::vector<int32_t const *>
    get_int32_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<int64_t const *>
    get_int64_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<float const *>
    get_float_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<double const *>
    get_double_ptrs(std::vector<GenericTensorAccessorR> const &);
std::vector<half const *>
    get_half_ptrs(std::vector<GenericTensorAccessorR> const &);

int32_t *get_int32_ptr(GenericTensorAccessorW const &);
int64_t *get_int64_ptr(GenericTensorAccessorW const &);
float *get_float_ptr(GenericTensorAccessorW const &);
double *get_double_ptr(GenericTensorAccessorW const &);
half *get_half_ptr(GenericTensorAccessorW const &);
std::vector<int32_t *>
    get_int32_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<int64_t *>
    get_int64_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<float *>
    get_float_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<double *>
    get_double_ptrs(std::vector<GenericTensorAccessorW> const &);
std::vector<half *> get_half_ptrs(std::vector<GenericTensorAccessorW> const &);

template <DataType DT>
std::vector<real_type_t<DT> const *>
    get(std::vector<GenericTensorAccessorR> const &accs) {
  std::vector<real_type_t<DT> const *> out;
  for (auto acc : accs) {
    out.push_back(get<DT>(acc));
  }
  return out;
}

GenericTensorAccessorR read_only_accessor_from_write_accessor(
    GenericTensorAccessorW const &write_accessor);

TensorShape get_tensor_shape_for_accessor_r(GenericTensorAccessorR const &);
TensorShape get_tensor_shape_for_accessor_w(GenericTensorAccessorW const &);

void copy_accessor_data_to_l_from_r(GenericTensorAccessorW const &dst_accessor,
                                    GenericTensorAccessorR const &src_accessor);

template <DataType DT>
real_type_t<DT> accessor_get_only_value(GenericTensorAccessorR const &acc) {
  ASSERT(get_num_elements(acc.shape.dims) == 1);
  ASSERT(acc.shape.data_type == DT);

  return *static_cast<real_type_t<DT> const *>(acc.ptr);
}

} // namespace FlexFlow

namespace FlexFlow {
static_assert(is_well_behaved_value_type_no_hash<GenericTensorAccessorR>::value,
              "");
static_assert(is_well_behaved_value_type_no_hash<GenericTensorAccessorW>::value,
              "");

} // namespace FlexFlow

#endif
