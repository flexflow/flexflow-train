#include "test_utils.h"
#include "kernels/datatype_dispatch.h"
#include "op-attrs/tensor_shape.h"
#include "utils/join_strings.h"
#include <random>

namespace FlexFlow {

TensorShape make_tensor_shape_from_ff_ordered(FFOrdered<nonnegative_int> dims,
                                              DataType DT) {
  return TensorShape{
      TensorDims{
          dims,
      },
      DT,
  };
}

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
      copy_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);
  return DataTypeDispatch1<CPUAccessorRContainsNonZero>{}(
      cpu_accessor.data_type, cpu_accessor);
}

template <DataType DT>
struct Print2DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    int const dims = accessor.shape.num_dims();
    int const cols = accessor.shape.at(legion_dim_t{0_n});
    int const rows = (dims == 2) ? accessor.shape.at(legion_dim_t{1_n}) : 1_n;

    auto get_element = [dims, &accessor](int j, int i) {
      return (dims == 1) ? accessor.at<DT>({j}) : accessor.at<DT>({j, i});
    };

    std::vector<int> indices(cols);
    std::iota(indices.begin(), indices.end(), 0);
    for (int i = 0; i < rows; ++i) {
      stream << join_strings(indices, " ", [=](int j) {
        return get_element(j, i);
      }) << std::endl;
    }
  }
};

void print_2d_tensor_accessor_contents(GenericTensorAccessorR const &accessor,
                                       std::ostream &stream) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      copy_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);
  DataTypeDispatch1<Print2DCPUAccessorR>{}(
      accessor.data_type, cpu_accessor, stream);
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
