#include "doctest/doctest.h"
#include "kernels/accessor.h"
#include "op-attrs/datatype_value.h"
#include "test_utils.h"

using namespace ::FlexFlow;

template <DataType DT>
void check_accessor_get(GenericTensorAccessorR const &accessor,
                        real_type_t<DT> expected) {
  CHECK(*accessor.get<DT>() == expected);

  if constexpr (DT == DataType::INT32) {
    CHECK(*accessor.get_int32_ptr() == expected);
  } else if constexpr (DT == DataType::INT64) {
    CHECK(*accessor.get_int64_ptr() == expected);
  } else if constexpr (DT == DataType::FLOAT) {
    CHECK(*accessor.get_float_ptr() == doctest::Approx(expected));
  } else if constexpr (DT == DataType::DOUBLE) {
    CHECK(*accessor.get_double_ptr() == doctest::Approx(expected));
  } else if constexpr (DT == DataType::HALF) {
    CHECK(*accessor.get_half_ptr() == doctest::Approx(expected));
  }
}

template <DataType DT>
void run_accessor_w_test(DataTypeValue value,
                         real_type_t<DT> expected,
                         Allocator allocator) {
  TensorShape shape = make_tensor_shape_from_ff_ordered({1_n}, DT);
  GenericTensorAccessorW accessor =
      create_filled_accessor_w(shape, allocator, value);
  check_accessor_get<DT>(read_only_accessor_from_write_accessor(accessor),
                         expected);
}

template <DataType DT>
void run_accessor_r_test(DataTypeValue value,
                         real_type_t<DT> expected,
                         Allocator allocator) {
  TensorShape shape = make_tensor_shape_from_ff_ordered({1_n}, DT);
  GenericTensorAccessorR accessor =
      create_filled_accessor_r(shape, allocator, value);
  check_accessor_get<DT>(accessor, expected);
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test GenericTensorAccessors") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("Test GenericTensorAccessorW") {
      SUBCASE("Test get methods for GenericTensorAccessorW") {
        run_accessor_w_test<DataType::INT32>(
            make_int32_data_type_value(12345), 12345, cpu_allocator);
        run_accessor_w_test<DataType::INT64>(
            make_int64_data_type_value(12345LL), 12345LL, cpu_allocator);
        run_accessor_w_test<DataType::FLOAT>(
            make_float_data_type_value(1.23f), 1.23f, cpu_allocator);
        run_accessor_w_test<DataType::DOUBLE>(
            make_double_data_type_value(1.23), 1.23, cpu_allocator);
      }

      SUBCASE("Test operator== and operator!= for GenericTensorAccessorW") {
        TensorShape shape =
            make_tensor_shape_from_ff_ordered({1_n}, DataType::INT32);

        GenericTensorAccessorW accessor1 = create_filled_accessor_w(
            shape, cpu_allocator, make_int32_data_type_value(12345));
        GenericTensorAccessorW accessor2 = create_filled_accessor_w(
            shape, cpu_allocator, make_int32_data_type_value(12345));
        GenericTensorAccessorW accessor3 = create_filled_accessor_w(
            shape, cpu_allocator, make_int32_data_type_value(54321));

        CHECK(accessor1 == accessor2);
        CHECK(accessor1 != accessor3);
      }

      SUBCASE("Test at() method for GenericTensorAccessorW") {
        DataType const DT = DataType::INT32;
        TensorShape shape = make_tensor_shape_from_ff_ordered({3_n, 3_n}, DT);

        GenericTensorAccessorW accessor_1 =
            create_random_filled_accessor_w(shape, cpu_allocator);
        GenericTensorAccessorW accessor_2 =
            copy_tensor_accessor_w(accessor_1, cpu_allocator);

        CHECK(accessor_1.at<DT>({0, 0}) == accessor_2.at<DT>({0, 0}));
        CHECK(accessor_1.at<DT>({1, 0}) == accessor_2.at<DT>({1, 0}));
        CHECK(accessor_1.at<DT>({2, 2}) == accessor_2.at<DT>({2, 2}));
      }
    }

    SUBCASE("Test GenericTensorAccessorR") {

      SUBCASE("Test get methods for GenericTensorAccessorR") {
        run_accessor_r_test<DataType::INT32>(
            make_int32_data_type_value(12345), 12345, cpu_allocator);
        run_accessor_r_test<DataType::INT64>(
            make_int64_data_type_value(12345LL), 12345LL, cpu_allocator);
        run_accessor_r_test<DataType::FLOAT>(
            make_float_data_type_value(1.23f), 1.23f, cpu_allocator);
        run_accessor_r_test<DataType::DOUBLE>(
            make_double_data_type_value(1.23), 1.23, cpu_allocator);
      }

      SUBCASE("Test operator== and operator!= for GenericTensorAccessorR") {
        TensorShape shape =
            make_tensor_shape_from_ff_ordered({1_n}, DataType::INT32);

        GenericTensorAccessorR accessor1 = create_filled_accessor_r(
            shape, cpu_allocator, make_int32_data_type_value(12345));
        GenericTensorAccessorR accessor2 = create_filled_accessor_r(
            shape, cpu_allocator, make_int32_data_type_value(12345));
        GenericTensorAccessorR accessor3 = create_filled_accessor_r(
            shape, cpu_allocator, make_int32_data_type_value(54321));

        CHECK(accessor1 == accessor2);
        CHECK(accessor1 != accessor3);
      }

      SUBCASE("Test at() method for GenericTensorAccessorR") {
        DataType const DT = DataType::INT32;
        TensorShape shape = make_tensor_shape_from_ff_ordered({3_n, 3_n}, DT);

        GenericTensorAccessorR accessor_1 =
            create_random_filled_accessor_r(shape, cpu_allocator);
        GenericTensorAccessorR accessor_2 =
            copy_tensor_accessor_r(accessor_1, cpu_allocator);

        CHECK(accessor_1.at<DT>({0, 0}) == accessor_2.at<DT>({0, 0}));
        CHECK(accessor_1.at<DT>({1, 0}) == accessor_2.at<DT>({1, 0}));
        CHECK(accessor_1.at<DT>({2, 2}) == accessor_2.at<DT>({2, 2}));
      }
    }
  }
}
