#include "op-attrs/task_space_coordinate.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("task_space_coord_set_is_orthotopic") {
    SUBCASE("coord set is empty") {
      std::set<TaskSpaceCoordinate> coord_set = {};

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set contains coords of different dimensionalities") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 3_n}}},
      };

      CHECK_THROWS(task_space_coord_set_is_orthotopic(coord_set));
    }

    SUBCASE("coord set contains only 0-d coords") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord set is not orthotopic") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 4_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{3_n, 0_n}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = false;

      CHECK(result == correct);
    }

    SUBCASE("coord set is orthotopic") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 1_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 1_n}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }

    SUBCASE("coord set's space is not minimal") {
      std::set<TaskSpaceCoordinate> coord_set = {
          TaskSpaceCoordinate{OrthotopeCoord{{0_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{1_n, 0_n}}},
          TaskSpaceCoordinate{OrthotopeCoord{{2_n, 0_n}}},
      };

      bool result = task_space_coord_set_is_orthotopic(coord_set);
      bool correct = true;

      CHECK(result == correct);
    }
  }

  TEST_CASE("task_coord_matching_parallel_tensor_space_coordinate") {
    ParallelTensorDimDegrees dim_degrees = ParallelTensorDimDegrees{
        /*sum_degree=*/SumDegree{4_p},
        /*discard_copy_degree=*/DiscardCopyDegree{6_p},
        /*shard_degrees=*/
        FFOrdered<positive_int>{
            9_p,
            1_p,
            8_p,
        },
    };

    SUBCASE("coord does not match matches degrees num dims") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/2_n,
          /*discard_copy_component=*/3_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              1_n,
              4_n,
          },
      };

      CHECK_THROWS(task_coord_matching_parallel_tensor_space_coordinate(
          coord, dim_degrees));
    }

    SUBCASE("coord is out-of-bounds for degrees") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/2_n,
          /*discard_copy_component=*/6_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              1_n,
              0_n,
              4_n,
          },
      };

      CHECK_THROWS(task_coord_matching_parallel_tensor_space_coordinate(
          coord, dim_degrees));
    }

    SUBCASE("coord is in bounds for degrees") {
      ParallelTensorSpaceCoordinate coord = ParallelTensorSpaceCoordinate{
          /*sum_component=*/2_n,
          /*discard_copy_component=*/3_n,
          /*shard_components=*/
          FFOrdered<nonnegative_int>{
              1_n,
              0_n,
              4_n,
          },
      };

      TaskSpaceCoordinate result =
          task_coord_matching_parallel_tensor_space_coordinate(coord,
                                                               dim_degrees);
      TaskSpaceCoordinate correct = TaskSpaceCoordinate{
          OrthotopeCoord{
              {2_n, 3_n, 1_n, 4_n},
          },
      };

      CHECK(result == correct);
    }
  }
}
