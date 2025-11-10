#include <doctest/doctest.h>
#include "utils/orthotope/dim_projection.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("compute_dim_projection") {
    std::string shard0_idx = "shard(1)";
    std::string shard1_idx = "shard(2)";
    std::string sum_idx = "sum";
    std::string discard_copy_idx = "discard_copy";

    DimOrdering<int> input_dim_ordering = DimOrdering<int>{
      [](int lhs, int rhs) {
        return lhs < rhs;
      },
    };

    DimOrdering<std::string> output_dim_ordering = DimOrdering<std::string>{
      [&](std::string const &lhs, std::string const &rhs) {
        auto numerate = [&](std::string const &s) {
          if (s == sum_idx) {
            return 0;
          } else if (s == discard_copy_idx) {
            return 1;
          } else if (s == shard0_idx) {
            return 2;
          } else if (s == shard1_idx) {
            return 3;
          } else {
            PANIC("Invalid index string", s);
          }
        };

        return numerate(lhs) < numerate(rhs);
      },
    };

    SUBCASE("up projection") {
      DimDomain<int> input_domain = 
        DimDomain<int>{{
          {0, 5_p},
          {1, 12_p},
        }};

      DimDomain<std::string> output_domain = 
        DimDomain<std::string>{{
          {sum_idx, 2_p},
          {discard_copy_idx, 1_p},
          {shard0_idx, 6_p},
          {shard1_idx, 5_p},
        }};

      DimCoord<int> input_coord = DimCoord<int>{{
        {0, 3_n}, 
        {1, 10_n},
      }};

      DimCoord<std::string> correct = DimCoord<std::string>{{
        {sum_idx, 1_n},
        {discard_copy_idx, 0_n},
        {shard0_idx, 4_n},
        {shard1_idx, 3_n},
      }};

      SUBCASE("all dims are mapped") {
        DimProjection<int, std::string> projection = [&] {
          UpProjection<int, std::string> 
            projection = make_empty_up_projection<int, std::string>();

          project_dims(
            projection, 
            /*onto=*/0,
            /*from=*/std::unordered_set{
              shard1_idx,
              discard_copy_idx,
            });
          project_dims(
            projection,
            /*onto=*/1,
            /*from=*/std::unordered_set{
              shard0_idx,
              sum_idx,
            });
          
          return DimProjection{
            projection,
          };
        }();

        DimCoord<std::string> result = 
          compute_dim_projection(
            /*projection=*/projection,
            /*input_coord=*/input_coord,
            /*input_domain=*/input_domain,
            /*output_domain=*/output_domain,
            /*input_dim_ordering=*/input_dim_ordering,
            /*output_dim_ordering=*/output_dim_ordering);

        CHECK(result == correct);
      }

      SUBCASE("all nontrivial dims are mapped") {
        DimProjection<int, std::string> projection = [&] {
          UpProjection<int, std::string> 
            projection = make_empty_up_projection<int, std::string>();

          project_dims(
            projection, 
            /*onto=*/0,
            /*from=*/std::unordered_set{
              shard1_idx,
            });
          project_dims(
            projection,
            /*onto=*/1,
            /*from=*/std::unordered_set{
              shard0_idx,
              sum_idx,
            });

          
          return DimProjection{
            projection,
          };
        }();

        DimCoord<std::string> result = 
          compute_dim_projection(
            /*projection=*/projection,
            /*input_coord=*/input_coord,
            /*input_domain=*/input_domain,
            /*output_domain=*/output_domain,
            /*input_dim_ordering=*/input_dim_ordering,
            /*output_dim_ordering=*/output_dim_ordering);

        CHECK(result == correct);
      }

      SUBCASE("not all nontrivial dims are mapped") {
        DimProjection<int, std::string> projection = [&] {
          UpProjection<int, std::string> 
            projection = make_empty_up_projection<int, std::string>();

          project_dims(
            projection, 
            /*onto=*/0,
            /*from=*/std::unordered_set{
              shard1_idx,
              discard_copy_idx,
            });
          project_dims(
            projection,
            /*onto=*/1,
            /*from=*/std::unordered_set{
              sum_idx,
            });

          
          return DimProjection{
            projection,
          };
        }();

        CHECK_THROWS(compute_dim_projection(
            /*projection=*/projection,
            /*input_coord=*/input_coord,
            /*input_domain=*/input_domain,
            /*output_domain=*/output_domain,
            /*input_dim_ordering=*/input_dim_ordering,
            /*output_dim_ordering=*/output_dim_ordering));
      }
    }
  }
}
