#include "kernels/format_accessor_contents.h"
#include "kernels/copy_tensor_accessor.h"
#include "kernels/datatype_dispatch.h"
#include "kernels/local_cpu_allocator.h"
#include "utils/indent.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <DataType DT>
struct Print1DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = accessor.shape.num_dims();
    ASSERT(dims == 1_n);

    nonnegative_int ncols = accessor.shape.at(ff_dim_t{0_n});

    stream << "["
           << join_strings(nonnegative_range(ncols),
                           " ",
                           [&](nonnegative_int col_idx) -> std::string {
                             return fmt::to_string(
                                 accessor.at<DT>(FFOrdered{col_idx}));
                           })
           << "]";
  }
};

static std::string
    format_1d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(accessor.shape.num_dims() == 1_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print1DCPUAccessorR>{}(accessor.data_type, accessor, oss);
  return oss.str();
}

template <DataType DT>
struct Print2DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = accessor.shape.num_dims();
    ASSERT(dims == 2_n);
    nonnegative_int dim0_size = accessor.shape.at(ff_dim_t{0_n});
    nonnegative_int dim1_size = accessor.shape.at(ff_dim_t{1_n});

    auto render_1d = [&](nonnegative_int dim0_idx) -> std::string {
      return "[" +
             join_strings(nonnegative_range(dim1_size),
                          " ",
                          [&](nonnegative_int dim1_idx) -> std::string {
                            return fmt::to_string(
                                accessor.at<DT>(FFOrdered{dim0_idx, dim1_idx}));
                          }) +
             "]";
    };

    stream << "[\n"
           << indent(
                  join_strings(nonnegative_range(dim0_size), "\n", render_1d))
           << "\n]";
  }
};

static std::string
    format_2d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(accessor.shape.num_dims() == 2_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print2DCPUAccessorR>{}(accessor.data_type, accessor, oss);
  return oss.str();
}

template <DataType DT>
struct Print3DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    ASSERT(accessor.device_type == DeviceType::CPU);
    nonnegative_int dims = accessor.shape.num_dims();
    ASSERT(dims == 3_n);

    nonnegative_int dim0_size = accessor.shape.at(ff_dim_t{0_n});
    nonnegative_int dim1_size = accessor.shape.at(ff_dim_t{1_n});
    nonnegative_int dim2_size = accessor.shape.at(ff_dim_t{2_n});

    auto render_1d = [&](nonnegative_int dim0_idx,
                         nonnegative_int dim1_idx) -> std::string {
      return "[" +
             join_strings(nonnegative_range(dim2_size),
                          " ",
                          [&](nonnegative_int dim2_idx) -> std::string {
                            return fmt::to_string(accessor.at<DT>(
                                FFOrdered{dim0_idx, dim1_idx, dim2_idx}));
                          }) +
             "]";
    };

    auto render_2d = [&](nonnegative_int dim0_idx) -> std::string {
      return "[\n" +
             indent(join_strings(nonnegative_range(dim1_size),
                                 "\n",
                                 [&](nonnegative_int dim1_idx) -> std::string {
                                   return render_1d(dim0_idx, dim1_idx);
                                 })) +
             "\n]";
    };

    stream << "[\n"
           << indent(
                  join_strings(nonnegative_range(dim0_size), "\n", render_2d))
           << "\n]";
  }
};

static std::string
    format_3d_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  ASSERT(accessor.device_type == DeviceType::CPU);
  ASSERT(accessor.shape.num_dims() == 3_n);

  std::ostringstream oss;
  DataTypeDispatch1<Print3DCPUAccessorR>{}(accessor.data_type, accessor, oss);
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

std::string format_accessor_r_contents(GenericTensorAccessorR const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      copy_tensor_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);

  int num_dims = accessor.shape.num_dims().unwrap_nonnegative();
  switch (num_dims) {
    case 1:
      return format_1d_accessor_r_contents(accessor);
    case 2:
      return format_2d_accessor_r_contents(accessor);
    case 3:
      return format_3d_accessor_r_contents(accessor);
    default:
      PANIC("Unhandled accessor dimensionality", num_dims);
  }
}

std::string format_accessor_w_contents(GenericTensorAccessorW const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor =
      copy_tensor_accessor_w_to_cpu_if_necessary(accessor, cpu_allocator);

  int num_dims = cpu_accessor.shape.num_dims().unwrap_nonnegative();
  switch (num_dims) {
    case 1:
      return format_1d_accessor_w_contents(cpu_accessor);
    case 2:
      return format_2d_accessor_w_contents(cpu_accessor);
    case 3:
      return format_3d_accessor_w_contents(cpu_accessor);
    default:
      PANIC("Unhandled accessor dimensionality", num_dims);
  }
}

} // namespace FlexFlow
