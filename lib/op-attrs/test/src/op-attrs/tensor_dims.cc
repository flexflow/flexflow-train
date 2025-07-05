#include "op-attrs/tensor_dims.h"
#include "test/utils/doctest/fmt/optional.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("tensor_dims_has_dim") {
    SUBCASE("nonempty tensor_dims") {
      TensorDims tensor_dims = TensorDims{FFOrdered{6_p, 9_p, 8_p}};

      SUBCASE("does have dim") {
        bool correct = true;
        SUBCASE("leading dim") {
          ff_dim_t dim = ff_dim_t{0_n};

          bool result = tensor_dims_has_dim(tensor_dims, dim);

          CHECK(result == correct);
        }

        SUBCASE("internal dim") {
          ff_dim_t dim = ff_dim_t{1_n};

          bool result = tensor_dims_has_dim(tensor_dims, dim);

          CHECK(result == correct);
        }

        SUBCASE("trailing dim") {
          ff_dim_t dim = ff_dim_t{2_n};

          bool result = tensor_dims_has_dim(tensor_dims, ff_dim_t{1_n});

          CHECK(result == correct);
        }
      }

      SUBCASE("dim is too large") {
        ff_dim_t dim = ff_dim_t{3_n};

        bool result = tensor_dims_has_dim(tensor_dims, dim);
        bool correct = false;

        CHECK(result == correct);
      }
    }

    SUBCASE("empty tensor_dims") {
      TensorDims tensor_dims = TensorDims{FFOrdered<positive_int>{}};
      ff_dim_t dim = ff_dim_t{0_n};

      bool result = tensor_dims_has_dim(tensor_dims, dim);
      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("tensor_dims_is_broadcastable_to(TensorDims, TensorDims)") {

    TensorDims goal = TensorDims{FFOrdered{1_p, 1_p, 4_p, 3_p}};

    SUBCASE("dims match") {
      bool result = tensor_dims_is_broadcastable_to(goal, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr only needs num_dims promotion") {
      TensorDims curr = TensorDims{FFOrdered{4_p, 3_p}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr only needs dim expansion") {
      TensorDims curr = TensorDims{FFOrdered{1_p, 1_p, 1_p, 3_p}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr needs both num_dims promotion and dim expansion") {
      TensorDims curr = TensorDims{FFOrdered{1_p, 3_p}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("curr needs invalid dim promotion") {
      TensorDims curr = TensorDims{FFOrdered{1_p, 1_p, 2_p, 3_p}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("num_dims(goal) < num_dims(curr)") {
      TensorDims curr = TensorDims{FFOrdered{1_p, 1_p, 10_p, 4_p, 3_p}};

      bool result = tensor_dims_is_broadcastable_to(curr, goal);
      bool correct = false;

      CHECK(result == correct);
    }
  }

  TEST_CASE("get_tensor_dims_coord_set") {
    SUBCASE("TensorDims is not empty") {
      TensorDims input = TensorDims{
          FFOrdered{3_p, 1_p, 2_p},
      };

      std::unordered_set<TensorDimsCoord> result = get_tensor_dims_coord_set(input);
      std::unordered_set<TensorDimsCoord> correct = {
          TensorDimsCoord{FFOrdered{0_n, 0_n, 0_n}},
          TensorDimsCoord{FFOrdered{0_n, 0_n, 1_n}},
          TensorDimsCoord{FFOrdered{1_n, 0_n, 0_n}},
          TensorDimsCoord{FFOrdered{1_n, 0_n, 1_n}},
          TensorDimsCoord{FFOrdered{2_n, 0_n, 0_n}},
          TensorDimsCoord{FFOrdered{2_n, 0_n, 1_n}},
      };

      CHECK(result == correct);
    }

    SUBCASE("TensorDims is zero-dimensional") {
      TensorDims input = TensorDims{FFOrdered<positive_int>{}};

      std::unordered_set<TensorDimsCoord> result = get_tensor_dims_coord_set(input);
      std::unordered_set<TensorDimsCoord> correct = {
          TensorDimsCoord{FFOrdered<nonnegative_int>{}},
      };

      CHECK(result == correct);
    }  
  }

  TEST_CASE("get_broadcast_target_dims(std::unordered_set<TensorDims>)") {
    TensorDims d1 = TensorDims{FFOrdered{1_p, 10_p, 4_p, 3_p}};

    TensorDims d2 = TensorDims{FFOrdered{10_p, 4_p, 1_p}};

    SUBCASE("has target in inputs") {
      TensorDims d3 = TensorDims{FFOrdered{1_p, 1_p, 4_p, 3_p}};

      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d2, d3});
      std::optional<TensorDims> correct = d1;

      CHECK(result == correct);
    }

    SUBCASE("has no possible target") {
      TensorDims d3 = TensorDims{FFOrdered{1_p, 1_p, 1_p, 4_p}};

      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d2, d3});
      std::optional<TensorDims> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("has possible target, but not in inputs") {
      TensorDims d3 = TensorDims{FFOrdered{1_p, 1_p, 1_p, 4_p, 3_p}};

      TensorDims possible_target =
          TensorDims{FFOrdered{1_p, 1_p, 10_p, 4_p, 3_p}};

      REQUIRE(tensor_dims_is_broadcastable_to(d1, possible_target));
      REQUIRE(tensor_dims_is_broadcastable_to(d2, possible_target));
      REQUIRE(tensor_dims_is_broadcastable_to(d3, possible_target));

      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d2, d3});
      std::optional<TensorDims> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("inputs is empty") {
      std::optional<TensorDims> result = get_broadcast_target_dims({});
      std::optional<TensorDims> correct = std::nullopt;

      CHECK(result == correct);
    }

    SUBCASE("all inputs are same") {
      std::optional<TensorDims> result =
          get_broadcast_target_dims({d1, d1, d1, d1, d1});
      std::optional<TensorDims> correct = d1;

      CHECK(result == correct);
    }
  }
}
