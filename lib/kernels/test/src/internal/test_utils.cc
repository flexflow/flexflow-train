#include "internal/test_utils.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/require_all_same1.h"
#include "utils/join_strings.h"
#include <random>

namespace FlexFlow {

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator) {
  GenericTensorAccessorW result_accessor = allocator.allocate_tensor(shape);
  fill_with_zeros(result_accessor);
  return result_accessor;
}

GenericTensorAccessorR create_zero_filled_accessor_r(TensorShape const &shape,
                                                     Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_zero_filled_accessor_w(shape, allocator);
  return read_only_accessor_from_write_accessor(accessor);
}

GenericTensorAccessorW
    create_1d_accessor_w_with_contents(std::vector<float> const &contents,
                                       Allocator &allocator) {
  nonnegative_int ncols = num_elements(contents);
  ASSERT(ncols > 0);

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{ncols}},
      DataType::FLOAT,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int col_idx : nonnegative_range(ncols)) {
    cpu_accessor.at<DataType::FLOAT>(FFOrdered{col_idx}) =
        contents.at(col_idx.unwrap_nonnegative());
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

GenericTensorAccessorW create_2d_accessor_w_with_contents(
    std::vector<std::vector<float>> const &contents, Allocator &allocator) {
  nonnegative_int nrows = num_elements(contents);
  ASSERT(nrows > 0);

  nonnegative_int ncols = throw_if_unexpected(
      require_all_same1(transform(contents, [](std::vector<float> const &row) {
        return num_elements(row);
      })));
  ASSERT(ncols > 0);

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{nrows, ncols}},
      DataType::FLOAT,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int row_idx : nonnegative_range(nrows)) {
    for (nonnegative_int col_idx : nonnegative_range(ncols)) {
      cpu_accessor.at<DataType::FLOAT>(FFOrdered{row_idx, col_idx}) =
          contents.at(row_idx.unwrap_nonnegative())
              .at(col_idx.unwrap_nonnegative());
    }
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

GenericTensorAccessorW create_3d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<float>>> const &contents,
    Allocator &allocator) {
  nonnegative_int dim0_size = num_elements(contents);
  ASSERT(dim0_size > 0);

  nonnegative_int dim1_size = throw_if_unexpected(require_all_same1(
      transform(contents, [](std::vector<std::vector<float>> const &m) {
        return num_elements(m);
      })));
  ASSERT(dim1_size > 0);

  nonnegative_int dim2_size = throw_if_unexpected(require_all_same1(
      transform(contents, [](std::vector<std::vector<float>> const &m) {
        return throw_if_unexpected(
            require_all_same1(transform(m, [](std::vector<float> const &vec) {
              return num_elements(vec);
            })));
      })));
  ASSERT(dim2_size > 0);

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{dim0_size, dim1_size, dim2_size}},
      DataType::FLOAT,
  };

  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorW cpu_accessor = cpu_allocator.allocate_tensor(shape);

  for (nonnegative_int dim0_idx : nonnegative_range(dim0_size)) {
    for (nonnegative_int dim1_idx : nonnegative_range(dim1_size)) {
      for (nonnegative_int dim2_idx : nonnegative_range(dim2_size)) {
        cpu_accessor.at<DataType::FLOAT>(
            FFOrdered{dim0_idx, dim1_idx, dim2_idx}) =
            contents.at(dim0_idx.unwrap_nonnegative())
                .at(dim1_idx.unwrap_nonnegative())
                .at(dim2_idx.unwrap_nonnegative());
      }
    }
  }

  GenericTensorAccessorW result = allocator.allocate_tensor(shape);
  copy_accessor_data_to_l_from_r(
      result, read_only_accessor_from_write_accessor(cpu_accessor));

  return result;
}

GenericTensorAccessorW create_4d_accessor_w_with_contents(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &contents,
    Allocator &allocator) {
  nonnegative_int dim0_size = num_elements(contents);
  ASSERT(dim0_size > 0);

  nonnegative_int dim1_size = throw_if_unexpected(require_all_same1(transform(
      contents, [](std::vector<std::vector<std::vector<float>>> const &t) {
        return num_elements(t);
      })));
  ASSERT(dim1_size > 0);

  nonnegative_int dim2_size = throw_if_unexpected(require_all_same1(transform(
      contents, [](std::vector<std::vector<std::vector<float>>> const &m) {
        return throw_if_unexpected(require_all_same1(
            transform(m, [](std::vector<std::vector<float>> const &vec) {
              return num_elements(vec);
            })));
      })));
  ASSERT(dim2_size > 0);

  nonnegative_int dim3_size = throw_if_unexpected(require_all_same1(transform(
      contents, [](std::vector<std::vector<std::vector<float>>> const &t) {
        return throw_if_unexpected(require_all_same1(
            transform(t, [](std::vector<std::vector<float>> const &mat) {
              return throw_if_unexpected(require_all_same1(
                  transform(mat, [](std::vector<float> const &vec) {
                    return num_elements(vec);
                  })));
            })));
      })));
  ASSERT(dim3_size > 0);

  TensorShape shape = TensorShape{
      TensorDims{FFOrdered{dim0_size, dim1_size, dim2_size, dim3_size}},
      DataType::FLOAT,
  };

  GenericTensorAccessorW accessor = allocator.allocate_tensor(shape);

  for (nonnegative_int dim0_idx : nonnegative_range(dim0_size)) {
    for (nonnegative_int dim1_idx : nonnegative_range(dim1_size)) {
      for (nonnegative_int dim2_idx : nonnegative_range(dim2_size)) {
        for (nonnegative_int dim3_idx : nonnegative_range(dim3_size)) {
          accessor.at<DataType::FLOAT>(
              FFOrdered{dim0_idx, dim1_idx, dim2_idx, dim3_idx}) =
              contents.at(dim0_idx.unwrap_nonnegative())
                  .at(dim1_idx.unwrap_nonnegative())
                  .at(dim2_idx.unwrap_nonnegative())
                  .at(dim3_idx.unwrap_nonnegative());
        }
      }
    }
  }

  return accessor;
}

GenericTensorAccessorR
    create_1d_accessor_r_with_contents(std::vector<float> const &contents,
                                       Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_1d_accessor_w_with_contents(contents, allocator));
}

GenericTensorAccessorR create_2d_accessor_r_with_contents(
    std::vector<std::vector<float>> const &contents, Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_2d_accessor_w_with_contents(contents, allocator));
}

GenericTensorAccessorR create_3d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<float>>> const &contents,
    Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_3d_accessor_w_with_contents(contents, allocator));
}

GenericTensorAccessorR create_4d_accessor_r_with_contents(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &contents,
    Allocator &allocator) {
  return read_only_accessor_from_write_accessor(
      create_4d_accessor_w_with_contents(contents, allocator));
}

template <DataType DT>
struct CreateRandomFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    using T = real_type_t<DT>;
    T *data_ptr = src_accessor.get<DT>();

    std::random_device rd;
    std::mt19937 gen(rd());
    size_t num_elements = get_num_elements(shape).unwrap_nonnegative();
    if constexpr (std::is_same<T, bool>::value) {
      std::bernoulli_distribution dist(0.5);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    } else if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<T> dist(-1.0, 1.0);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    } else if constexpr (std::is_integral<T>::value) {
      std::uniform_int_distribution<T> dist(0, 99);
      for (size_t i = 0; i < num_elements; i++) {
        data_ptr[i] = dist(gen);
      }
    }

    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);
    copy_accessor_data_to_l_from_r(dst_accessor, src_accessor);

    return dst_accessor;
  }
};

GenericTensorAccessorW create_random_filled_accessor_w(TensorShape const &shape,
                                                       Allocator &allocator) {
  return DataTypeDispatch1<CreateRandomFilledAccessorW>{}(
      shape.data_type, shape, allocator);
}

GenericTensorAccessorR create_random_filled_accessor_r(TensorShape const &shape,
                                                       Allocator &allocator) {
  GenericTensorAccessorW accessor =
      create_random_filled_accessor_w(shape, allocator);

  return read_only_accessor_from_write_accessor(accessor);
}

template <DataType DT>
struct FillWithZeros {
  void operator()(GenericTensorAccessorW const &accessor) {
    using T = real_type_t<DT>;

    if (accessor.device_type == DeviceType::CPU) {
      memset(accessor.ptr,
             0,
             accessor.shape.get_volume().unwrap_nonnegative() * sizeof(T));
    } else {
      checkCUDA(cudaMemset(accessor.ptr,
                           0,
                           accessor.shape.get_volume().unwrap_nonnegative() *
                               sizeof(T)));
    }
  }
};

void fill_with_zeros(GenericTensorAccessorW const &accessor) {
  DataTypeDispatch1<FillWithZeros>{}(accessor.data_type, accessor);
}

template <DataType DT>
struct CPUAccessorRContainsNonZero {
  bool operator()(GenericTensorAccessorR const &accessor) {
    using T = real_type_t<DT>;

    T const *data_ptr = accessor.get<DT>();

    int volume = accessor.shape.num_elements().unwrap_nonnegative();
    for (size_t i = 0; i < volume; i++) {
      if (data_ptr[i] != 0) {
        return true;
      }
    }

    return false;
  }
};

bool contains_non_zero(GenericTensorAccessorR const &accessor) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      copy_tensor_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);
  return DataTypeDispatch1<CPUAccessorRContainsNonZero>{}(
      cpu_accessor.data_type, cpu_accessor);
}

template <DataType DT>
struct AccessorsAreEqual {
  bool operator()(GenericTensorAccessorR const &accessor_a,
                  GenericTensorAccessorR const &accessor_b) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorR cpu_accessor_a =
        copy_tensor_accessor_r_to_cpu_if_necessary(accessor_a, cpu_allocator);
    GenericTensorAccessorR cpu_accessor_b =
        copy_tensor_accessor_r_to_cpu_if_necessary(accessor_b, cpu_allocator);

    using T = real_type_t<DT>;
    T const *a_data_ptr = cpu_accessor_a.get<DT>();
    T const *b_data_ptr = cpu_accessor_b.get<DT>();

    int volume = accessor_a.shape.num_elements().unwrap_nonnegative();
    for (size_t i = 0; i < volume; i++) {
      if (a_data_ptr[i] != b_data_ptr[i]) {
        return false;
      }
    }

    return true;
  }
};

bool accessors_are_equal(GenericTensorAccessorR const &accessor_a,
                         GenericTensorAccessorR const &accessor_b) {
  ASSERT(accessor_a.shape == accessor_b.shape,
         "accessors_are_equal expects accessors to have the same shape");

  return DataTypeDispatch1<AccessorsAreEqual>{}(
      accessor_a.data_type, accessor_a, accessor_b);
}

template <DataType DT>
struct CreateFilledAccessorW {
  GenericTensorAccessorW operator()(TensorShape const &shape,
                                    Allocator &allocator,
                                    DataTypeValue val) {
    using T = real_type_t<DT>;
    if (!val.template has<T>()) {
      throw mk_runtime_error("create_filed_accessor expected data type of "
                             "shape and passed-in value to match");
    }

    auto unwrapped_value = val.get<T>();
    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorW src_accessor = cpu_allocator.allocate_tensor(shape);

    T *data_ptr = src_accessor.get<DT>();

    int volume = dst_accessor.shape.num_elements().unwrap_nonnegative();
    for (size_t i = 0; i < volume; i++) {
      data_ptr[i] = unwrapped_value;
    }

    copy_accessor_data_to_l_from_r(dst_accessor, src_accessor);
    return dst_accessor;
  }
};

GenericTensorAccessorW create_filled_accessor_w(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val) {

  return DataTypeDispatch1<CreateFilledAccessorW>{}(
      shape.data_type, shape, allocator, val);
}

GenericTensorAccessorR create_filled_accessor_r(TensorShape const &shape,
                                                Allocator &allocator,
                                                DataTypeValue val) {
  GenericTensorAccessorW w_accessor =
      create_filled_accessor_w(shape, allocator, val);
  return read_only_accessor_from_write_accessor(w_accessor);
}
} // namespace FlexFlow
