#include "kernels/format_accessor_contents.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "op-attrs/tensor_shape.h"
#include "utils/indent.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <DataType DT>
struct Print1DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = get_num_dims(accessor.shape.dims).nonnegative_int_from_num_tensor_dims();
    ASSERT(dims == 1_n);

    positive_int ncols = dim_at_idx(accessor.shape.dims, ff_dim_t{0_n});

    stream << "["
           << join_strings(
                  nonnegative_range(ncols.nonnegative_int_from_positive_int()),
                  " ",
                  [&](nonnegative_int col_idx) -> std::string {
                    return fmt::to_string(
                        accessor.at<DT>(TensorDimsCoord{FFOrdered{col_idx}}));
                  })
           << "]";
  }
};

static std::string
    format_1d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(get_num_dims(accessor.shape.dims) == 1_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print1DCPUAccessorR>{}(
      accessor.shape.data_type, accessor, oss);
  return oss.str();
}

template <DataType DT>
struct Print2DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = get_num_dims(accessor.shape.dims).nonnegative_int_from_num_tensor_dims();
    ASSERT(dims == 2_n);
    positive_int dim0_size = dim_at_idx(accessor.shape.dims, ff_dim_t{0_n});
    positive_int dim1_size = dim_at_idx(accessor.shape.dims, ff_dim_t{1_n});

    auto render_1d = [&](nonnegative_int dim0_idx) -> std::string {
      return "[" +
             join_strings(
                 nonnegative_range(
                     dim1_size.nonnegative_int_from_positive_int()),
                 " ",
                 [&](nonnegative_int dim1_idx) -> std::string {
                   return fmt::to_string(accessor.at<DT>(
                       TensorDimsCoord{FFOrdered{dim0_idx, dim1_idx}}));
                 }) +
             "]";
    };

    stream << "[\n"
           << indent(join_strings(
                  nonnegative_range(
                      dim0_size.nonnegative_int_from_positive_int()),
                  "\n",
                  render_1d))
           << "\n]";
  }
};

static std::string
    format_2d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(get_num_dims(accessor.shape.dims) == 2_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print2DCPUAccessorR>{}(
      accessor.shape.data_type, accessor, oss);
  return oss.str();
}

template <DataType DT>
struct Print3DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = get_num_dims(accessor.shape.dims).nonnegative_int_from_num_tensor_dims();
    ASSERT(dims == 3_n);

    positive_int dim0_size = dim_at_idx(accessor.shape.dims, ff_dim_t{0_n});
    positive_int dim1_size = dim_at_idx(accessor.shape.dims, ff_dim_t{1_n});
    positive_int dim2_size = dim_at_idx(accessor.shape.dims, ff_dim_t{2_n});

    auto render_1d = [&](nonnegative_int dim0_idx,
                         nonnegative_int dim1_idx) -> std::string {
      return "[" +
             join_strings(nonnegative_range(
                              dim2_size.nonnegative_int_from_positive_int()),
                          " ",
                          [&](nonnegative_int dim2_idx) -> std::string {
                            return fmt::to_string(
                                accessor.at<DT>(TensorDimsCoord{
                                    FFOrdered{dim0_idx, dim1_idx, dim2_idx}}));
                          }) +
             "]";
    };

    auto render_2d = [&](nonnegative_int dim0_idx) -> std::string {
      return "[\n" +
             indent(join_strings(
                 nonnegative_range(
                     dim1_size.nonnegative_int_from_positive_int()),
                 "\n",
                 [&](nonnegative_int dim1_idx) -> std::string {
                   return render_1d(dim0_idx, dim1_idx);
                 })) +
             "\n]";
    };

    stream << "[\n"
           << indent(join_strings(
                  nonnegative_range(
                      dim0_size.nonnegative_int_from_positive_int()),
                  "\n",
                  render_2d))
           << "\n]";
  }
};

static std::string
    format_3d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(get_num_dims(accessor.shape.dims) == 3_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print3DCPUAccessorR>{}(
      accessor.shape.data_type, accessor, oss);
  return oss.str();
}

template <DataType DT>
struct Print4DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = get_num_dims(accessor.shape.dims).nonnegative_int_from_num_tensor_dims();
    ASSERT(dims == 4_n);

    positive_int dim0_size = dim_at_idx(accessor.shape.dims, ff_dim_t{0_n});
    positive_int dim1_size = dim_at_idx(accessor.shape.dims, ff_dim_t{1_n});
    positive_int dim2_size = dim_at_idx(accessor.shape.dims, ff_dim_t{2_n});
    positive_int dim3_size = dim_at_idx(accessor.shape.dims, ff_dim_t{3_n});

    auto render_1d = [&](nonnegative_int dim0_idx,
                         nonnegative_int dim1_idx,
                         nonnegative_int dim2_idx) -> std::string {
      return "[" +
             join_strings(
                 nonnegative_range(
                     dim3_size.nonnegative_int_from_positive_int()),
                 " ",
                 [&](nonnegative_int dim3_idx) -> std::string {
                   return fmt::to_string(accessor.at<DT>(TensorDimsCoord{
                       FFOrdered{dim0_idx, dim1_idx, dim2_idx, dim3_idx}}));
                 }) +
             "]";
    };

    auto render_2d = [&](nonnegative_int dim0_idx,
                         nonnegative_int dim1_idx) -> std::string {
      return "[\n" +
             indent(join_strings(
                 nonnegative_range(
                     dim2_size.nonnegative_int_from_positive_int()),
                 "\n",
                 [&](nonnegative_int dim2_idx) -> std::string {
                   return render_1d(dim0_idx, dim1_idx, dim2_idx);
                 })) +
             "\n]";
    };

    auto render_3d = [&](nonnegative_int dim0_idx) -> std::string {
      return "[\n" +
             indent(join_strings(
                 nonnegative_range(
                     dim1_size.nonnegative_int_from_positive_int()),
                 "\n",
                 [&](nonnegative_int dim1_idx) -> std::string {
                   return render_2d(dim0_idx, dim1_idx);
                 })) +
             "\n]";
    };

    stream << "[\n"
           << indent(join_strings(
                  nonnegative_range(
                      dim0_size.nonnegative_int_from_positive_int()),
                  "\n",
                  render_3d))
           << "\n]";
  }
};

static std::string
    format_4d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(get_num_dims(accessor.shape.dims) == 4_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print4DCPUAccessorR>{}(
      accessor.shape.data_type, accessor, oss);
  return oss.str();
}

static std::string
    format_1d_accessor_w_contents(GenericTensorAccessorW const &accessor) {
  return format_1d_accessor_r_contents(
      read_only_accessor_from_write_accessor(accessor));
}

static std::string
    format_2d_accessor_w_contents(GenericTensorAccessorW const &accessor) {
  return format_2d_accessor_r_contents(
      read_only_accessor_from_write_accessor(accessor));
}

static std::string
    format_3d_accessor_w_contents(GenericTensorAccessorW const &accessor) {
  return format_3d_accessor_r_contents(
      read_only_accessor_from_write_accessor(accessor));
}

static std::string
    format_4d_accessor_w_contents(GenericTensorAccessorW const &accessor) {
  return format_4d_accessor_r_contents(
      read_only_accessor_from_write_accessor(accessor));
}

std::string format_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      copy_tensor_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);

  int num_dims = get_num_dims(cpu_accessor.shape.dims).int_from_num_tensor_dims();
  switch (num_dims) {
    case 1:
      return format_1d_accessor_r_contents(cpu_accessor);
    case 2:
      return format_2d_accessor_r_contents(cpu_accessor);
    case 3:
      return format_3d_accessor_r_contents(cpu_accessor);
    case 4:
      return format_4d_accessor_r_contents(cpu_accessor);
    default:
      PANIC("Unhandled accessor dimensionality", num_dims);
  }
}

std::string format_accessor_w_contents(GenericTensorAccessorW const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor =
      copy_tensor_accessor_w_to_cpu_if_necessary(accessor, cpu_allocator);

  int num_dims = get_num_dims(cpu_accessor.shape.dims).int_from_num_tensor_dims();
  switch (num_dims) {
    case 1:
      return format_1d_accessor_w_contents(cpu_accessor);
    case 2:
      return format_2d_accessor_w_contents(cpu_accessor);
    case 3:
      return format_3d_accessor_w_contents(cpu_accessor);
    case 4:
      return format_4d_accessor_w_contents(cpu_accessor);
    default:
      PANIC("Unhandled accessor dimensionality", num_dims);
  }
}

} // namespace FlexFlow
