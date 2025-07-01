#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "kernels/array_shape.h"
#include "kernels/device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "pcg/device_type.dtg.h"
#include "utils/containers/transform.h"
#include <libassert/assert.hpp>
#include <string>

namespace FlexFlow {

nonnegative_int
    calculate_accessor_offset(LegionOrdered<nonnegative_int> const &,
                              ArrayShape const &);

class GenericTensorAccessorR {
public:
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type const *get() const {
    ASSERT(this->data_type == DT, "Invalid datatype requested");

    return static_cast<real_type_t<DT> const *>(this->ptr);
  }

  int32_t const *get_int32_ptr() const;
  int64_t const *get_int64_ptr() const;
  float const *get_float_ptr() const;
  double const *get_double_ptr() const;
  half const *get_half_ptr() const;

  GenericTensorAccessorR() = delete;

  GenericTensorAccessorR(DataType data_type,
                         ArrayShape const &shape,
                         void const *ptr,
                         DeviceType device_type);

  bool operator==(GenericTensorAccessorR const &) const;
  bool operator!=(GenericTensorAccessorR const &) const;

  template <DataType DT>
  real_type_t<DT> const &at(FFOrdered<nonnegative_int> const &indices) const {
    return this->at<DT>(legion_ordered_from_ff_ordered(indices));
  }

  template <DataType DT>
  real_type_t<DT> const &
      at(LegionOrdered<nonnegative_int> const &indices) const {
    ASSERT(this->device_type == DeviceType::CPU,
           "GenericTensorAccessorR::at() requires CPU-allocated tensor");
    ASSERT(this->data_type == DT, "Invalid datatype requested");

    using T = real_type_t<DT>;
    T const *data_ptr = static_cast<T const *>(this->ptr);
    nonnegative_int offset = calculate_accessor_offset(indices, this->shape);
    return data_ptr[offset.unwrap_nonnegative()];
  }

public:
  DataType data_type;
  ArrayShape shape;
  void const *ptr;
  DeviceType device_type;

private:
  std::tuple<decltype(data_type) const &,
             decltype(shape) const &,
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
    ASSERT(this->data_type == DT, "Invalid datatype requested");

    return static_cast<real_type_t<DT> *>(this->ptr);
  }

  int32_t *get_int32_ptr() const;
  int64_t *get_int64_ptr() const;
  float *get_float_ptr() const;
  double *get_double_ptr() const;
  half *get_half_ptr() const;

  GenericTensorAccessorW() = delete;

  GenericTensorAccessorW(DataType data_type,
                         ArrayShape const &shape,
                         void *ptr,
                         DeviceType device_type);

  bool operator==(GenericTensorAccessorW const &) const;
  bool operator!=(GenericTensorAccessorW const &) const;

  operator GenericTensorAccessorR() const;

  template <DataType DT>
  real_type_t<DT> &at(FFOrdered<nonnegative_int> const &indices) {
    return this->at<DT>(legion_ordered_from_ff_ordered(indices));
  }

  template <DataType DT>
  real_type_t<DT> &at(LegionOrdered<nonnegative_int> const &indices) {
    ASSERT(this->device_type == DeviceType::CPU,
           "GenericTensorAccessorW::at() requires CPU-allocated tensor");
    ASSERT(this->data_type == DT, "Invalid datatype requested");

    using T = real_type_t<DT>;
    T *data_ptr = static_cast<T *>(this->ptr);
    nonnegative_int offset = calculate_accessor_offset(indices, this->shape);
    return data_ptr[offset.unwrap_nonnegative()];
  }

  template <DataType DT>
  real_type_t<DT> &at(FFOrdered<nonnegative_int> const &indices) const {
    return this->at<DT>(legion_ordered_from_ff_ordered(indices));
  }

  template <DataType DT>
  real_type_t<DT> &at(LegionOrdered<nonnegative_int> const &indices) const {
    ASSERT(this->device_type == DeviceType::CPU,
           "GenericTensorAccessorW::at() requires CPU-allocated tensor");
    ASSERT(this->data_type == DT, "Invalid datatype requested");

    using T = real_type_t<DT>;
    T *data_ptr = static_cast<T *>(this->ptr);
    nonnegative_int offset = calculate_accessor_offset(indices, this->shape);
    return data_ptr[offset.unwrap_nonnegative()];
  }

public:
  DataType data_type;
  ArrayShape shape;
  void *ptr;
  DeviceType device_type;

private:
  std::tuple<decltype(data_type) const &,
             decltype(shape) const &,
             decltype(ptr) const &,
             decltype(device_type) const &>
      tie() const;
};

std::string format_as(GenericTensorAccessorW const &);
std::ostream &operator<<(std::ostream &, GenericTensorAccessorW const &);

template <DataType DT>
typename data_type_enum_to_class<DT>::type *
    get(GenericTensorAccessorW const &a) {
  ASSERT(a.data_type == DT, "Invalid datatype requested");
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
  ASSERT(a.data_type == DT, "Invalid datatype requested");
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

bool is_shape_and_dtype_equal(GenericTensorAccessorR const &acc1,
                              GenericTensorAccessorR const &acc2);
bool is_shape_and_dtype_equal(GenericTensorAccessorW const &acc1,
                              GenericTensorAccessorW const &acc2);

bool shape_and_dtype_matches(GenericTensorAccessorR const &accessor,
                             ArrayShape const &expected_shape,
                             DataType const &expected_dtype);
bool shape_and_dtype_matches(GenericTensorAccessorW const &accessor,
                             ArrayShape const &expected_shape,
                             DataType const &expected_dtype);

std::pair<ArrayShape, DataType>
    get_shape_and_datatype(GenericTensorAccessorR const &accessor);
std::pair<ArrayShape, DataType>
    get_shape_and_datatype(GenericTensorAccessorW const &accessor);

TensorShape get_tensor_shape_for_accessor_r(GenericTensorAccessorR const &);
TensorShape get_tensor_shape_for_accessor_w(GenericTensorAccessorW const &);

void copy_accessor_data_to_l_from_r(GenericTensorAccessorW &dst_accessor,
                                    GenericTensorAccessorR const &src_accessor);

template <DataType DT>
real_type_t<DT> accessor_get_only_value(GenericTensorAccessorR const &acc) {
  ASSERT(get_num_elements(acc.shape) == 1);
  ASSERT(acc.data_type == DT);

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
