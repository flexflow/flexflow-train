#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/orthotope/up_projection.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static parallel_tensor_dim_idx_t shard_dim_idx_from_raw(int idx) {
  return parallel_tensor_dim_idx_t{ff_dim_t{nonnegative_int{idx}}};
}

static operator_task_space_dim_idx_t op_task_space_dim_from_raw(int idx) {
  return operator_task_space_dim_idx_t{nonnegative_int{idx}};
}

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_identity_mapping(ParallelTensorDimDegrees)") {
    nonnegative_int num_shard_dims = nonnegative_int{2};

    OperatorSpaceToParallelTensorSpaceMapping result =
        get_identity_mapping(num_shard_dims);

    OperatorSpaceToParallelTensorSpaceMapping correct =
        OperatorSpaceToParallelTensorSpaceMapping{
            DimProjection{
                EqProjection{
                    bidict<operator_task_space_dim_idx_t,
                           parallel_tensor_dim_idx_t>{
                        {
                            op_task_space_dim_from_raw(0),
                            shard_dim_idx_from_raw(0),
                        },
                        {
                            op_task_space_dim_from_raw(1),
                            shard_dim_idx_from_raw(1),
                        },
                        {
                            op_task_space_dim_from_raw(2),
                            parallel_tensor_dim_idx_t{
                                ReplicaType::DISCARD_COPY},
                        },
                        {
                            op_task_space_dim_from_raw(3),
                            parallel_tensor_dim_idx_t{ReplicaType::SUM},
                        },
                    },
                },
            },
        };

    CHECK(result == correct);
  }

  TEST_CASE("ptensor_coord_for_task_space_coord") {
    SUBCASE("up projection") {
      OperatorSpaceToParallelTensorSpaceMapping mapping = [&] {
        UpProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> 
          projection = make_empty_up_projection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>();

        project_dims(
          projection, 
          /*onto=*/operator_task_space_dim_idx_t{0_n},
          /*from=*/std::unordered_set{
            shard_dim_idx(ff_dim_t{1_n}),
            discard_copy_dim_idx(),
          });
        project_dims(
          projection,
          /*onto=*/operator_task_space_dim_idx_t{1_n},
          /*from=*/std::unordered_set{
            shard_dim_idx(ff_dim_t{0_n}),
            sum_dim_idx(),
          });

        
        return OperatorSpaceToParallelTensorSpaceMapping{{
            DimProjection{
              projection,
            },
          }};
      }();

      OperatorTaskSpace op_task_space = OperatorTaskSpace{
        std::vector<positive_int>{
          5_p,
          12_p,
        },
      };

      ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{2_p},
        /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        /*shard_degrees=*/FFOrdered{
          6_p,
          5_p,
        },
      };

      TaskSpaceCoordinate task_space_coordinate = TaskSpaceCoordinate{
        OrthotopeCoord{
          std::vector{
            3_n,
            10_n,
          },
        },
      };

      ParallelTensorSpaceCoordinate result = 
        ptensor_coord_for_task_space_coord(
          /*mapping=*/mapping,
          /*op_task_space=*/op_task_space,
          /*ptensor_dim_degrees=*/dim_degrees,
          /*task_space_coord=*/task_space_coordinate);

      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
        /*sum_component=*/1_n,
        /*discard_copy_component=*/0_n,
        /*shard_components=*/FFOrdered{
          4_n,
          3_n,
        },
      };

      CHECK(result == correct);
    }

    SUBCASE("identity projection") {
      nonnegative_int num_shard_dims = 2_n;
      OperatorSpaceToParallelTensorSpaceMapping mapping = get_identity_mapping(num_shard_dims);

      OperatorTaskSpace op_task_space = OperatorTaskSpace{
        std::vector<positive_int>{
          5_p,
          3_p,
          12_p,
          2_p,
        },
      };

      ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{5_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/FFOrdered{
          12_p,
          2_p,
        },
      };

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
          /*op_task_space=*/op_task_space,
          /*ptensor_dim_degrees=*/dim_degrees,
          /*task_space_coord=*/task_space_coordinate);

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

    SUBCASE("some dims unmapped") {
      OperatorSpaceToParallelTensorSpaceMapping mapping = [&] {
        UpProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> 
          projection = make_empty_up_projection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>();

        project_dims(
          projection, 
          /*onto=*/operator_task_space_dim_idx_t{0_n},
          /*from=*/std::unordered_set{
            shard_dim_idx(ff_dim_t{1_n}),
            discard_copy_dim_idx(),
          });
        project_dims(
          projection,
          /*onto=*/operator_task_space_dim_idx_t{1_n},
          /*from=*/std::unordered_set{
            shard_dim_idx(ff_dim_t{0_n}),
            sum_dim_idx(),
          });

        
        return OperatorSpaceToParallelTensorSpaceMapping{{
            DimProjection{
              projection,
            },
          }};
      }();

      OperatorTaskSpace op_task_space = OperatorTaskSpace{
        std::vector<positive_int>{
          12_p,
        },
      };

      ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{4_p},
        /*discard_copy_degree=*/DiscardCopyDegree{3_p},
        /*shard_degrees=*/FFOrdered{
          1_p,
          1_p,
        },
      };

      TaskSpaceCoordinate task_space_coordinate = TaskSpaceCoordinate{
        OrthotopeCoord{
          std::vector{
            10_n,
          },
        },
      };

      ParallelTensorSpaceCoordinate result = 
        ptensor_coord_for_task_space_coord(
          /*mapping=*/mapping,
          /*op_task_space=*/op_task_space,
          /*ptensor_dim_degrees=*/dim_degrees,
          /*task_space_coord=*/task_space_coordinate);

      ParallelTensorSpaceCoordinate correct = ParallelTensorSpaceCoordinate{
        /*sum_component=*/3_n,
        /*discard_copy_component=*/1_n,
        /*shard_components=*/FFOrdered{
          1_n,
          1_n,
        },
      };

      CHECK(result == correct);
      
    }
  }

  TEST_CASE("task_space_coord_for_ptensor_coord") {
    SUBCASE("up projection") {
      OperatorSpaceToParallelTensorSpaceMapping mapping = [&] {
        UpProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> 
          projection = make_empty_up_projection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t>();

        project_dims(
          projection, 
          /*onto=*/operator_task_space_dim_idx_t{0_n},
          /*from=*/std::unordered_set{
            shard_dim_idx(ff_dim_t{1_n}),
            discard_copy_dim_idx(),
          });
        project_dims(
          projection,
          /*onto=*/operator_task_space_dim_idx_t{1_n},
          /*from=*/std::unordered_set{
            shard_dim_idx(ff_dim_t{0_n}),
            sum_dim_idx(),
          });

        
        return OperatorSpaceToParallelTensorSpaceMapping{{
            DimProjection{
              projection,
            },
          }};
      }();

      OperatorTaskSpace op_task_space = OperatorTaskSpace{
        std::vector<positive_int>{
          5_p,
          12_p,
        },
      };

      ParallelTensorSpaceCoordinate tensor_space_coordinate  = ParallelTensorSpaceCoordinate{
        /*sum_component=*/1_n,
        /*discard_copy_component=*/0_n,
        /*shard_components=*/FFOrdered{
          4_n,
          3_n,
        },
      };

      ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{2_p},
        /*discard_copy_degree=*/DiscardCopyDegree{1_p},
        /*shard_degrees=*/FFOrdered{
          6_p,
          5_p,
        },
      };

      TaskSpaceCoordinate result = 
        task_space_coord_for_ptensor_coord(
          /*mapping=*/mapping,
          /*ptensor_dim_degrees=*/dim_degrees,
          /*op_task_space=*/op_task_space,
          /*task_space_coord=*/tensor_space_coordinate);

      TaskSpaceCoordinate correct = TaskSpaceCoordinate{
        OrthotopeCoord{
          std::vector{
            3_n,
            10_n,
          },
        },
      };


      CHECK(result == correct);
    }
  }
}
