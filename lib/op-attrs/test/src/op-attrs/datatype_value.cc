#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>
#include "test/utils/doctest/fmt/half.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("test make_data_type_value") {
    SUBCASE("make_half_data_type_value") {
      half value = 3.0f;
      DataTypeValue data_type_value = make_half_data_type_value(value);

      CHECK(data_type_value.has<half>());
      CHECK_FALSE(data_type_value.has<float>());
      CHECK_FALSE(data_type_value.has<double>());
      CHECK_FALSE(data_type_value.has<int32_t>());
      CHECK_FALSE(data_type_value.has<int64_t>());
      CHECK_FALSE(data_type_value.has<bool>());
      CHECK(data_type_value.get<half>() == value);
    }

    SUBCASE("make_float_data_type_value") {
      float value = 1.0f;
      DataTypeValue data_type_value = make_float_data_type_value(value);

      CHECK(data_type_value.has<float>());
      CHECK_FALSE(data_type_value.has<half>());
      CHECK_FALSE(data_type_value.has<double>());
      CHECK_FALSE(data_type_value.has<int32_t>());
      CHECK_FALSE(data_type_value.has<int64_t>());
      CHECK_FALSE(data_type_value.has<bool>());
      CHECK(data_type_value.get<float>() == value);
    }

    SUBCASE("make_double_data_type_value") {
      double value = 2.71828;
      DataTypeValue data_type_value = make_double_data_type_value(value);

      CHECK(data_type_value.has<double>());
      CHECK_FALSE(data_type_value.has<half>());
      CHECK_FALSE(data_type_value.has<float>());
      CHECK_FALSE(data_type_value.has<int32_t>());
      CHECK_FALSE(data_type_value.has<int64_t>());
      CHECK_FALSE(data_type_value.has<bool>());
      CHECK(data_type_value.get<double>() == value);
    }

    SUBCASE("make_int32_data_type_value") {
      int32_t value = -42;
      DataTypeValue data_type_value = make_int32_data_type_value(value);

      CHECK(data_type_value.has<int32_t>());
      CHECK_FALSE(data_type_value.has<half>());
      CHECK_FALSE(data_type_value.has<float>());
      CHECK_FALSE(data_type_value.has<double>());
      CHECK_FALSE(data_type_value.has<int64_t>());
      CHECK_FALSE(data_type_value.has<bool>());
      CHECK(data_type_value.get<int32_t>() == value);
    }

    SUBCASE("make_int64_data_type_value") {
      int64_t value = 1LL << 40;
      DataTypeValue data_type_value = make_int64_data_type_value(value);

      CHECK(data_type_value.has<int64_t>());
      CHECK_FALSE(data_type_value.has<half>());
      CHECK_FALSE(data_type_value.has<float>());
      CHECK_FALSE(data_type_value.has<double>());
      CHECK_FALSE(data_type_value.has<int32_t>());
      CHECK_FALSE(data_type_value.has<bool>());
      CHECK(data_type_value.get<int64_t>() == value);
    }

    SUBCASE("make_bool_data_type_value") {
      bool value = true;
      DataTypeValue data_type_value = make_bool_data_type_value(value);

      CHECK(data_type_value.has<bool>());
      CHECK_FALSE(data_type_value.has<half>());
      CHECK_FALSE(data_type_value.has<float>());
      CHECK_FALSE(data_type_value.has<double>());
      CHECK_FALSE(data_type_value.has<int32_t>());
      CHECK_FALSE(data_type_value.has<int64_t>());
      CHECK(data_type_value.get<bool>() == value);
    }
  }

  TEST_CASE("get_data_type_of_data_type_value") {
    SUBCASE("half") {
      DataTypeValue input = make_half_data_type_value(0.0);
       
      DataType result = get_data_type_of_data_type_value(input);
      DataType correct = DataType::HALF;

      CHECK(result == correct);
    } 

    SUBCASE("float") {
      DataTypeValue input = make_float_data_type_value(0.0);
       
      DataType result = get_data_type_of_data_type_value(input);
      DataType correct = DataType::FLOAT;

      CHECK(result == correct);
    }

    SUBCASE("double") {
      DataTypeValue input = make_double_data_type_value(0.0);
       
      DataType result = get_data_type_of_data_type_value(input);
      DataType correct = DataType::DOUBLE;

      CHECK(result == correct);
    }

    SUBCASE("int32") {
      DataTypeValue input = make_int32_data_type_value(0);
       
      DataType result = get_data_type_of_data_type_value(input);
      DataType correct = DataType::INT32;

      CHECK(result == correct);
    }

    SUBCASE("int64") {
      DataTypeValue input = make_int64_data_type_value(0);
       
      DataType result = get_data_type_of_data_type_value(input);
      DataType correct = DataType::INT64;

      CHECK(result == correct);
    }

    SUBCASE("bool") {
      DataTypeValue input = make_bool_data_type_value(false); 

      DataType result = get_data_type_of_data_type_value(input);
      DataType correct = DataType::BOOL;

      CHECK(result == correct);
    }
  }
}
