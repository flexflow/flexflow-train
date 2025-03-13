#include "doctest/doctest.h"
#include "kernels/array_shape.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test ArrayShape") {
    ArrayShape shape({1_n, 2_n, 3_n, 4_n});

    SUBCASE("Test get_volume() and num_elements()") {
      CHECK(shape.get_volume() == 1 * 2 * 3 * 4);
      CHECK(shape.num_elements() == 1 * 2 * 3 * 4);
    }

    SUBCASE("Test num_dims() and get_dim()") {
      CHECK(shape.num_dims() == 4);
      CHECK(shape.get_dim() == 4);
    }

    SUBCASE("Test operator[] and at()") {
      CHECK(shape[legion_dim_t{0_n}] == 1);
      CHECK(shape[legion_dim_t{1_n}] == 2);
      CHECK(shape[legion_dim_t{2_n}] == 3);
      CHECK(shape[legion_dim_t{3_n}] == 4);

      CHECK(shape.at(legion_dim_t{0_n}) == 1);
      CHECK(shape.at(legion_dim_t{1_n}) == 2);
      CHECK(shape.at(legion_dim_t{2_n}) == 3);
      CHECK(shape.at(legion_dim_t{3_n}) == 4);

      CHECK(shape.at(ff_dim_t{0_n}) == 4);
      CHECK(shape.at(ff_dim_t{1_n}) == 3);
      CHECK(shape.at(ff_dim_t{2_n}) == 2);
      CHECK(shape.at(ff_dim_t{3_n}) == 1);
    }

    SUBCASE("Test operator== and operator!=") {
      ArrayShape shape2({1_n, 2_n, 3_n, 4_n});
      ArrayShape shape3({1_n, 2_n, 3_n, 5_n});

      CHECK(shape == shape2);
      CHECK(shape != shape3);
    }

    SUBCASE("Test last_idx()") {
      CHECK(shape.last_idx() == legion_dim_t{3_n});

      ArrayShape empty_shape(std::vector<nonnegative_int>{});
      CHECK_THROWS(empty_shape.last_idx());
    }

    SUBCASE("Test neg_idx()") {
      CHECK(shape.neg_idx(-1) == legion_dim_t{3_n});
      CHECK(shape.neg_idx(-2) == legion_dim_t{2_n});
      CHECK(shape.neg_idx(-3) == legion_dim_t{1_n});
      CHECK(shape.neg_idx(-4) == legion_dim_t{0_n});

      CHECK_THROWS(shape.neg_idx(-5));
    }

    SUBCASE("Test at_maybe()") {
      CHECK(shape.at_maybe(legion_dim_t{0_n}).value() == 1);
      CHECK(shape.at_maybe(legion_dim_t{1_n}).value() == 2);
      CHECK(shape.at_maybe(legion_dim_t{2_n}).value() == 3);
      CHECK(shape.at_maybe(legion_dim_t{3_n}).value() == 4);
      CHECK(!shape.at_maybe(legion_dim_t{4_n}).has_value());

      CHECK(shape.at_maybe(ff_dim_t{0_n}).value() == 4);
      CHECK(shape.at_maybe(ff_dim_t{1_n}).value() == 3);
      CHECK(shape.at_maybe(ff_dim_t{2_n}).value() == 2);
      CHECK(shape.at_maybe(ff_dim_t{3_n}).value() == 1);
      CHECK(!shape.at_maybe(ff_dim_t{4_n}).has_value());
    }

    SUBCASE("Test subshape()") {
      SUBCASE("Test basic subshape") {
        ArrayShape ref_shape({2_n, 3_n});
        ArrayShape subshape =
            shape.sub_shape(legion_dim_t{1_n}, legion_dim_t{3_n});

        CHECK(ref_shape == subshape);
      }

      SUBCASE("Test empty subshape") {
        ArrayShape ref_shape(std::vector<nonnegative_int>{});
        ArrayShape subshape =
            shape.sub_shape(legion_dim_t{0_n}, legion_dim_t{0_n});
        CHECK(ref_shape == subshape);
      }

      SUBCASE("Test subshape with no start") {
        ArrayShape ref_shape({1_n, 2_n, 3_n});
        ArrayShape subshape = shape.sub_shape(std::nullopt, legion_dim_t{3_n});
        CHECK(ref_shape == subshape);
      }

      SUBCASE("Test subshape with no end") {
        ArrayShape ref_shape({2_n, 3_n, 4_n});
        ArrayShape subshape = shape.sub_shape(legion_dim_t{1_n}, std::nullopt);
        CHECK(ref_shape == subshape);
      }
    }
  }
}
