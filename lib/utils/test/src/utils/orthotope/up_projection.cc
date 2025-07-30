#include <doctest/doctest.h>
#include "utils/orthotope/dim_ordering.h"
#include "utils/orthotope/up_projection.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compute_up_projection") {
    SUBCASE("input domain has 1 dim") {
      DimCoord<int> coord = DimCoord<int>{{
        {3, 7_n},
      }}; 

      DimDomain<std::string> output_domain = DimDomain<std::string>{{
        {"a", 3_p},
        {"b", 4_p},
      }};

      UpProjection<int, std::string> projection = UpProjection{
        OneToMany<int, std::string>{
          {3, {"a", "b"}},
        },
      };

      SUBCASE("dim ordering 1") {
        DimOrdering<std::string> output_dim_ordering = make_dim_ordering_from_vector<std::string>({
          "b", "a",
        });

        DimCoord<std::string> result = compute_up_projection(projection, coord, output_domain, output_dim_ordering);
        DimCoord<std::string> correct = DimCoord<std::string>{{
          {"a", 1_n},
          {"b", 2_n},
        }};

        CHECK(result == correct);
      }

      SUBCASE("dim ordering 2") {
        DimOrdering<std::string> output_dim_ordering = make_dim_ordering_from_vector<std::string>({
          "a", "b",
        });

        DimCoord<std::string> result = compute_up_projection(projection, coord, output_domain, output_dim_ordering);
        DimCoord<std::string> correct = DimCoord<std::string>{{
          {"a", 1_n},
          {"b", 3_n},
        }};

        CHECK(result == correct);
      }
    }

    SUBCASE("input domain has multiple dims") {
      DimCoord<int> coord = DimCoord<int>{{
        {3, 9_n},
        {1, 2_n},
      }}; 

      DimDomain<std::string> output_domain = DimDomain<std::string>{{
        {"a", 3_p},
        {"b", 4_p},
        {"c", 5_p},
      }};

      UpProjection<int, std::string> projection = UpProjection{
        OneToMany<int, std::string>{
          {3, {"a", "c"}},
          {1, {"b"}},
        },
      };

      DimOrdering<std::string> output_dim_ordering = make_dim_ordering_from_vector<std::string>({
        "a", "b", "c",
      });

      DimCoord<std::string> result = compute_up_projection(projection, coord, output_domain, output_dim_ordering);
      DimCoord<std::string> correct = DimCoord<std::string>{{
        {"a", 1_n},
        {"b", 2_n},
        {"c", 4_n},
      }};

      CHECK(result == correct);
    }
  }
}
