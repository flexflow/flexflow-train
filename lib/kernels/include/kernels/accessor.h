#ifndef _FLEXFLOW_KERNELS_ACCESSOR_H
#define _FLEXFLOW_KERNELS_ACCESSOR_H

#include "array_shape.h"
#include "device.h"
#include "kernels/ff_handle.h"
#include "op-attrs/datatype.h"
#include "pcg/device_type.dtg.h"
#include "utils/exception.h"
#include "utils/required.h"

namespace FlexFlow {

inline int calculate_accessor_offset(std::vector<int> const &indices,
                                     ArrayShape const &shape) {
  int offset = 0;
  int multiplier = 1;

  for (int i = 0; i < shape.num_dims(); i++) {
    if (indices.at(i) >= shape.at(legion_dim_t{nonnegative_int{i}})) {
      throw mk_runtime_error(
          fmt::format("In {} dimension, attempting to access index {} "
                      "when only {} indexes exist",
                      i,
                      indices.at(i),
                      shape.at(legion_dim_t{nonnegative_int{i}})));
    }

    offset += indices.at(i) * multiplier;
    multiplier *=
        shape.at(legion_dim_t{nonnegative_int{i}}).unwrap_nonnegative();
  }

  return offset;
}

class GenericTensorAccessorR {
public:
  template <DataType DT>
  typename data_type_enum_to_class<DT>::type const *get() const {
    if (this->data_type == DT) {
      return static_cast<real_type_t<DT> const *>(this->ptr);
    } else {
      throw mk_runtime_error(fmt::format(
          "Invalid access data type ({} != {})", this->data_type, DT));
    }
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
  real_type_t<DT> const &at(std::vector<int> const &indices) const {
    if (this->device_type != DeviceType::CPU) {
      throw mk_runtime_error("Calling at() on non-CPU allocated tensor");
    }
    if (this->data_type != DT) {
      throw mk_runtime_error(fmt::format(
          "Invalid access data type ({} != {})", this->data_type, DT));
    }
    if (indices.size() != this->shape.num_dims()) {
      throw mk_runtime_error(fmt::format("Number of indices ({}) does not "
                                         "match the number of dimensions ({}).",
                                         indices.size(),
                                         this->shape.num_dims()));
    }

    using T = real_type_t<DT>;
    T const *data_ptr = static_cast<T const *>(this->ptr);
    int offset = calculate_accessor_offset(indices, this->shape);
    return data_ptr[offset];
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
    if (this->data_type == DT) {
      return static_cast<real_type_t<DT> *>(this->ptr);
    } else {
      throw mk_runtime_error(fmt::format(
          "Invalid access data type ({} != {})", this->data_type, DT));
    }
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
  real_type_t<DT> &at(std::vector<int> const &indices) {
    if (this->device_type != DeviceType::CPU) {
      throw mk_runtime_error("Calling at() on non-CPU allocated tensor");
    }
    if (this->data_type != DT) {
      throw mk_runtime_error(fmt::format(
          "Invalid access data type ({} != {})", this->data_type, DT));
    }
    if (indices.size() != this->shape.num_dims()) {
      throw mk_runtime_error(fmt::format("Number of indices ({}) does not "
                                         "match the number of dimensions ({}).",
                                         indices.size(),
                                         this->shape.num_dims()));
    }

    using T = real_type_t<DT>;
    T *data_ptr = static_cast<T *>(this->ptr);
    int offset = calculate_accessor_offset(indices, this->shape);
    return data_ptr[offset];
  }

  template <DataType DT>
  real_type_t<DT> &at(std::vector<int> const &indices) const {
    if (this->device_type != DeviceType::CPU) {
      throw mk_runtime_error("Calling at() on non-CPU allocated tensor");
    }
    if (this->data_type != DT) {
      throw mk_runtime_error(fmt::format(
          "Invalid access data type ({} != {})", this->data_type, DT));
    }
    if (indices.size() != this->shape.num_dims()) {
      throw mk_runtime_error(fmt::format("Number of indices ({}) does not "
                                         "match the number of dimensions ({}).",
                                         indices.size(),
                                         this->shape.num_dims()));
    }

    using T = real_type_t<DT>;
    T const *data_ptr = static_cast<T const *>(this->ptr);
    int offset = calculate_accessor_offset(indices, this->shape);
    return data_ptr[offset];
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

static_assert(is_fmtable<req<DataType> const &>::value, "");

template <DataType DT>
typename data_type_enum_to_class<DT>::type *
    get(GenericTensorAccessorW const &a) {
  if (a.data_type == DT) {
    return static_cast<real_type_t<DT> *>(a.ptr);
  } else {
    throw mk_runtime_error(
        fmt::format("Invalid access data type ({} != {})", a.data_type, DT));
  }
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
  if (a.data_type == DT) {
    return static_cast<real_type_t<DT> const *>(a.ptr);
  } else {
    throw mk_runtime_error(
        fmt::format("Invalid access data type ({} != {})", a.data_type, DT));
  }
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

bool shape_and_dtype_matches(GenericTensorAccessorR const &accessor,
                             ArrayShape const &expected_shape,
                             DataType const &expected_dtype);

std::pair<ArrayShape, DataType>
    get_shape_and_datatype(GenericTensorAccessorR const &accessor);

void copy_accessor_data_to_l_from_r(GenericTensorAccessorW &dst_accessor,
                                    GenericTensorAccessorR const &src_accessor);

} // namespace FlexFlow

namespace FlexFlow {
static_assert(is_well_behaved_value_type_no_hash<GenericTensorAccessorR>::value,
              "");
static_assert(is_well_behaved_value_type_no_hash<GenericTensorAccessorW>::value,
              "");

} // namespace FlexFlow

#endif
