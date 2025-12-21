#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/orthotope/up_projection.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_identity_mapping(OperatorTaskSpace, ParallelTensorDimDegrees)") {
    ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
      /*sum_degree=*/SumDegree{2_p},
      /*discard_copy_degree=*/DiscardCopyDegree{1_p},
      /*shard_degrees=*/FFOrdered{
        1_p,
        3_p,
        1_p,
      },
    };

    OperatorTaskSpace operator_task_space = OperatorTaskSpace{MinimalOrthotope{{
      3_ge2, 2_ge2,
    }}};

    OperatorSpaceToParallelTensorSpaceMapping result =
        get_identity_mapping(operator_task_space, dim_degrees);

    auto make_op_coord = [](nonnegative_int x, nonnegative_int y) {
      return DimCoord<operator_task_space_dim_idx_t>{{
        {operator_task_space_dim_idx_t{0_n}, x},  
        {operator_task_space_dim_idx_t{1_n}, y},
      }};
    };

    auto make_pt_coord = [](nonnegative_int sum_coord_entry, nonnegative_int shard_coord_entry) {
      return DimCoord<parallel_tensor_dim_idx_t>{{
        {sum_dim_idx(), sum_coord_entry},
        {discard_copy_dim_idx(), 0_n},
        {shard_dim_idx(ff_dim_t{0_n}), 0_n},
        {shard_dim_idx(ff_dim_t{1_n}), shard_coord_entry},
        {shard_dim_idx(ff_dim_t{2_n}), 0_n},
      }};
    };

    OperatorSpaceToParallelTensorSpaceMapping correct =
        OperatorSpaceToParallelTensorSpaceMapping{
          DimDomainMapping<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>{
            /*coord_mapping=*/bidict<
              DimCoord<operator_task_space_dim_idx_t>, 
              DimCoord<parallel_tensor_dim_idx_t>
            >{
              {make_op_coord(0_n, 0_n), make_pt_coord(0_n, 0_n)},
              {make_op_coord(0_n, 1_n), make_pt_coord(1_n, 0_n)},
              {make_op_coord(1_n, 0_n), make_pt_coord(0_n, 1_n)},
              {make_op_coord(1_n, 1_n), make_pt_coord(1_n, 1_n)},
              {make_op_coord(2_n, 0_n), make_pt_coord(0_n, 2_n)},
              {make_op_coord(2_n, 1_n), make_pt_coord(1_n, 2_n)},
            },
            /*l_domain=*/DimDomain<operator_task_space_dim_idx_t>{{
              {operator_task_space_dim_idx_t{0_n}, 3_p},   
              {operator_task_space_dim_idx_t{1_n}, 2_p},
            }},
            /*r_domain=*/DimDomain<parallel_tensor_dim_idx_t>{{
              {sum_dim_idx(), 2_p},
              {discard_copy_dim_idx(), 1_p},
              {shard_dim_idx(ff_dim_t{0_n}), 1_p},
              {shard_dim_idx(ff_dim_t{1_n}), 3_p},
              {shard_dim_idx(ff_dim_t{2_n}), 1_p},
            }},
        },
    };

    CHECK(result == correct);
  }

  TEST_CASE("ptensor_coord_for_task_space_coord") {
    SUBCASE("identity projection") {
      OperatorTaskSpace op_task_space = OperatorTaskSpace{
        MinimalOrthotope{{
          5_ge2,
          3_ge2,
          12_ge2,
          2_ge2,
        }},
      };

      ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{5_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/FFOrdered{
          12_p,
          2_p,
        },
      };
       
      OperatorSpaceToParallelTensorSpaceMapping mapping = get_identity_mapping(op_task_space, dim_degrees);

      TaskSpaceCoordinate task_space_coordinate = TaskSpaceCoordinate{
        OrthotopeCoord{
          std::vector{
            3_n,
            2_n,
            10_n,
            1_n,
          },
        },
      };

      ParallelTensorSpaceCoordinate result = 
        ptensor_coord_for_task_space_coord(
          /*mapping=*/mapping,
          /*task_space_coord=*/task_space_coordinate,
          /*num_dims=*/num_ptensor_shard_dims_t{2_n});

      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
        /*sum_component=*/3_n,
        /*discard_copy_component=*/2_n,
        /*shard_components=*/FFOrdered{
          10_n,
          1_n,
        },
      };

      CHECK(result == correct);
    }
  }
}
