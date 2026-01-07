#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/dim_ordering.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compute_down_projection") {
    SUBCASE("output domain has 1 dim") {
      DimCoord<std::string> coord = DimCoord<std::string>{{
          {"a", 1_n},
          {"b", 2_n},
      }};

      DimDomain<std::string> input_domain = DimDomain<std::string>{{
          {"a", 3_p},
          {"b", 4_p},
      }};

      DownProjection<std::string, int> projection = DownProjection{
          ManyToOne<std::string, int>{
              {{"a", "b"}, 3},
          },
      };

      SUBCASE("dim ordering 1") {
        DimOrdering<std::string> input_dim_ordering =
            make_dim_ordering_from_vector<std::string>({
                "b",
                "a",
            });

        DimCoord<int> result = compute_down_projection(
            projection, coord, input_domain, input_dim_ordering);
        DimCoord<int> correct = DimCoord<int>{{
            {3, 7_n},
        }};

        CHECK(result == correct);
      }

      SUBCASE("dim ordering 2") {
        DimOrdering<std::string> input_dim_ordering =
            make_dim_ordering_from_vector<std::string>({
                "a",
                "b",
            });

        DimCoord<int> result = compute_down_projection(
            projection, coord, input_domain, input_dim_ordering);
        DimCoord<int> correct = DimCoord<int>{{
            {3, 6_n},
        }};

        CHECK(result == correct);
      }
    }

    SUBCASE("not all output dims are mapped") {
      DimOrdering<std::string> input_dim_ordering =
          make_dim_ordering_from_vector<std::string>({
              "a",
              "b",
          });

      DownProjection<std::string, int> projection = make_empty_down_projection<std::string, int>();
      project_dims(projection, {"a", "b"}, 3);
      CHECK_THROWS(project_dims(projection, {}, 4));
   }
  }
}
