#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/datatype_dispatch.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

nonnegative_int
    calculate_accessor_offset(LegionOrdered<nonnegative_int> const &indices,
                              ArrayShape const &shape) {
  ASSERT(indices.size() == shape.num_dims(),
         "Number of indices does not match the number of dimensions");

  nonnegative_int offset = 0_n;
  nonnegative_int multiplier = 1_n;

  for (legion_dim_t dim : key_range(shape.dims)) {
    ASSERT(indices.at(dim) < shape.at(legion_dim_t{dim}),
           "Out of bounds access",
           dim);

    offset += indices.at(dim) * multiplier;
    multiplier *= shape.at(legion_dim_t{dim});
  }

  return offset;
}

void copy_accessor_data_to_l_from_r(
    GenericTensorAccessorW &dst_accessor,
    GenericTensorAccessorR const &src_accessor) {
  size_t num_bytes =
      dst_accessor.shape.get_volume().unwrap_nonnegative() *
      size_of_datatype(dst_accessor.data_type).unwrap_nonnegative();

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
  return fmt::format("<GenericTensorAccessorW data_type={} shape={} ptr={}>",
                     a.data_type,
                     a.shape,
                     a.ptr);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorW const &a) {
  return (s << fmt::to_string(a));
}

template <DataType DT>
struct Print2DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = accessor.shape.num_dims();
    ASSERT(dims == 2_n);
    nonnegative_int ncols = accessor.shape.at(legion_dim_t{0_n});
    nonnegative_int nrows = accessor.shape.at(legion_dim_t{1_n});

    auto render_row = [&](nonnegative_int row_idx) {
      stream << "[ ";
      for (nonnegative_int col_idx : nonnegative_range(ncols)) {
        stream << accessor.at<DT>(LegionOrdered{col_idx, row_idx}) << " ";
      }
      stream << "]" << std::endl;
    };

    for (nonnegative_int row_idx : nonnegative_range(nrows)) {
      render_row(row_idx);
    }
  }
};

std::string
    format_2d_accessor_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(accessor.shape.num_dims() == 2_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print2DCPUAccessorR>{}(accessor.data_type, accessor, oss);
  return oss.str();
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
  return fmt::format("<GenericTensorAccessorR data_type={} shape={} ptr={}>",
                     a.data_type,
                     a.shape,
                     a.ptr);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorR const &a) {
  return (s << fmt::to_string(a));
}

std::string
    format_2d_accessor_contents(GenericTensorAccessorW const &accessor) {
  return format_2d_accessor_contents(
      read_only_accessor_from_write_accessor(accessor));
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
