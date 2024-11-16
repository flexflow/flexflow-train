#include "test_utils.h"
#include "op-attrs/tensor_shape.h"
#include <random>

namespace FlexFlow {

GenericTensorAccessorW create_zero_filled_accessor_w(TensorShape const &shape,
                                                     Allocator &allocator) {
  GenericTensorAccessorW result_accessor = allocator.allocate_tensor(shape);
  fill_with_zeros(result_accessor);
  return result_accessor;
}

TensorShape
    make_tensor_shape_from_legion_dims(LegionOrdered<size_t> const &dims,
                                       DataType DT) {
  return TensorShape{
      TensorDims{
          ff_ordered_from_legion_ordered(dims),
      },
      DT,
  };
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
    size_t num_elements = get_num_elements(shape);
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
      std::uniform_int_distribution<T> dist(0, 100);
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
      memset(accessor.ptr, 0, accessor.shape.get_volume() * sizeof(T));
    } else {
      checkCUDA(
          cudaMemset(accessor.ptr, 0, accessor.shape.get_volume() * sizeof(T)));
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

    for (size_t i = 0; i < accessor.shape.num_elements(); i++) {
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

GenericTensorAccessorR
    copy_accessor_r_to_cpu_if_necessary(GenericTensorAccessorR const &accessor,
                                        Allocator &cpu_allocator) {
  GenericTensorAccessorR cpu_accessor = accessor;
  if (accessor.device_type == DeviceType::GPU) {
    cpu_accessor = copy_tensor_accessor_r(accessor, cpu_allocator);
  }
  return cpu_accessor;
}

GenericTensorAccessorW
    copy_accessor_w_to_cpu_if_necessary(GenericTensorAccessorW const &accessor,
                                        Allocator &cpu_allocator) {
  GenericTensorAccessorW cpu_accessor = accessor;
  if (accessor.device_type == DeviceType::GPU) {
    cpu_accessor = copy_tensor_accessor_w(accessor, cpu_allocator);
  }
  return cpu_accessor;
}

template <DataType DT>
struct Print2DCPUAccessorR {
  void operator()(GenericTensorAccessorR const &accessor,
                  std::ostream &stream) {
    using T = real_type_t<DT>;

    T const *data_ptr = accessor.get<DT>();
    int rows = accessor.shape.at(legion_dim_t{0});
    int cols = accessor.shape.at(legion_dim_t{1});

    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        stream << data_ptr[i * cols + j];

        if (j < cols - 1) {
          stream << " ";
        }
      }
      stream << std::endl;
    }
  }
};

void print_2d_tensor_accessor_contents(GenericTensorAccessorR const &accessor,
                                       std::ostream &stream) {
  Allocator cpu_allocator = create_local_cpu_memory_allocator();
  GenericTensorAccessorR cpu_accessor =
      copy_accessor_r_to_cpu_if_necessary(accessor, cpu_allocator);
  DataTypeDispatch1<Print2DCPUAccessorR>{}(
      accessor.data_type, accessor, stream);
}

template <DataType DT>
struct AccessorsAreEqual {
  bool operator()(GenericTensorAccessorR const &accessor_a,
                  GenericTensorAccessorR const &accessor_b) {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();
    GenericTensorAccessorR cpu_accessor_a =
        copy_accessor_r_to_cpu_if_necessary(accessor_a, cpu_allocator);
    GenericTensorAccessorR cpu_accessor_b =
        copy_accessor_r_to_cpu_if_necessary(accessor_b, cpu_allocator);

    using T = real_type_t<DT>;
    T const *a_data_ptr = cpu_accessor_a.get<DT>();
    T const *b_data_ptr = cpu_accessor_b.get<DT>();

    for (size_t i = 0; i < accessor_a.shape.num_elements(); i++) {
      if (a_data_ptr[i] != b_data_ptr[i]) {
        return false;
      }
    }

    return true;
  }
};

bool accessors_are_equal(GenericTensorAccessorR const &accessor_a,
                         GenericTensorAccessorR const &accessor_b) {
  if (accessor_a.shape != accessor_b.shape) {
    throw mk_runtime_error(
        fmt::format("accessors_are_equal expected accessors to have the same "
                    "shape, but received: {} != {}",
                    accessor_a.shape,
                    accessor_b.shape));
  }
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
    for (size_t i = 0; i < dst_accessor.shape.num_elements(); i++) {
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
