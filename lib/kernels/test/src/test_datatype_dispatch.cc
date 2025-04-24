#include "doctest/doctest.h"
#include "kernels/datatype_dispatch.h"

using namespace ::FlexFlow;

template <DataType DT>
struct TestDatatypeDispatch1 {
  int operator()(int value) {
    if (DT == DataType::FLOAT) {
      return value + 1;
    } else if (DT == DataType::INT32) {
      return value + 2;
    } else {
      return value + 3;
    }
  }
};

template <DataType IDT, DataType ODT>
struct TestDatatypeDispatch2 {
  void operator()(int &value) {
    if (IDT == DataType::INT32 && ODT == DataType::FLOAT) {
      value *= 2;
    } else if (IDT == DataType::FLOAT && ODT == DataType::INT32) {
      value *= 3;
    } else {
      value *= 4;
    }
  }
};

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test DataTypeDispatch") {
    SUBCASE("Test DataTypeDispatch1") {
      CHECK(DataTypeDispatch1<TestDatatypeDispatch1>{}(DataType::FLOAT, 1) ==
            2);
      CHECK(DataTypeDispatch1<TestDatatypeDispatch1>{}(DataType::INT32, 1) ==
            3);
      CHECK(DataTypeDispatch1<TestDatatypeDispatch1>{}(DataType::DOUBLE, 1) ==
            4);
    }

    SUBCASE("Test DataTypeDispatch2") {
      int value = 1;

      SUBCASE("Case One") {
        DataTypeDispatch2<TestDatatypeDispatch2>{}(
            DataType::INT32, DataType::FLOAT, value);
        CHECK(value == 2);
      }

      SUBCASE("Case Two") {
        DataTypeDispatch2<TestDatatypeDispatch2>{}(
            DataType::FLOAT, DataType::INT32, value);
        CHECK(value == 3);
      }

      SUBCASE("Test Three") {
        DataTypeDispatch2<TestDatatypeDispatch2>{}(
            DataType::DOUBLE, DataType::DOUBLE, value);
        CHECK(value == 4);
      }
    }
  }
}
