#include "compiler/machine_mapping/machine_view.h"
#include "op-attrs/ff_ordered/ff_ordered.h"
#include "op-attrs/task_space_coordinate.h"
#include "pcg/gpu_id_t.dtg.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/containers/transform.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("mv_get_expected_task_space_num_dims") {
    MachineView mv = MachineView{
        MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/0_n,
        },
        {
            MachineViewDimension{
                stride_t{2_p},
                MachineSpecificationDimension::INTER_NODE,
            },
            MachineViewDimension{
                stride_t{2_p},
                MachineSpecificationDimension::INTER_NODE,
            },
        },
    };

    CHECK(mv_get_expected_task_space_num_dims(mv) == 2_n);
  }

  TEST_CASE("get_machine_space_coordinate") {
    SUBCASE("1D case") {
      /**
       * This operator has shape (3,), and thus 3 tasks.
       * The (only) dimension is projected on the INTER (device) dimension with
       * a stride of 2. The start of the projection defined by MachineView
       * starts at MachineSpaceCoordinate (0,1), and the machine space has 1
       * node and 6 devices per node.
       *
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+
       *  |       | (0,)  |       | (1,)  |       | (2,)  |
       *  +-------+-------+-------+-------+-------+-------+
       * Where the (x,) are the `TaskSpaceCoordinate`s, and the underlying grid
       * is the machine space.
       */
      OperatorTaskSpace task = OperatorTaskSpace{
          MinimalOrthotope{{
              3_ge2,
          }},
      };

      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/0_n,
              /*device_idx=*/1_n,
          },
          {
              MachineViewDimension{
                  stride_t{2_p},
                  MachineSpecificationDimension::INTRA_NODE,
              },
          },
      };

      SUBCASE("Task with TaskSpaceCoordinate = (0,)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n});

        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);

        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/1_n,
        };

        CHECK(result == correct);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n});

        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);

        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/3_n,
        };

        CHECK(result == correct);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (2,)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({2_n});

        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);

        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0_n,
            /*device_idx=*/5_n,
        };

        CHECK(result == correct);
      }

      SUBCASE("TaskSpaceCoordinate is out of bounds") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({4_n});

        CHECK_THROWS(get_machine_space_coordinate(task, mv, coord));
      }
    }

    SUBCASE("2D case - projection on different dimensions") {
      /**
       * This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
       * The first dimension is projected onto the INTER (node) dimension with
       * stride 1, while the second dimension is projected onto the INTRA
       * (device) dimension with stride 2. The start of the projection defined
       * by MachineView is at MachineSpaceCoordinates (1, 2), and the machine
       * space has 3 nodes and 5 devices per node.
       *
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+
       *  |       |       |       |       |       |
       *  +-------+-------+-------+-------+-------+
       *  |       |       | (0,0) |       | (0,1) |
       *  +-------+-------+-------+-------+-------+
       *  |       |       | (1,0) |       | (1,1) |
       *  +-------+-------+-------+-------+-------+
       * Where the (x,y) are the `TaskSpaceCoordinate`s, and the underlying
       * grid is the machine space.
       */

      OperatorTaskSpace task = OperatorTaskSpace{
          MinimalOrthotope{{
              2_ge2,
              2_ge2,
          }},
      };
      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/1_n,
              /*device_idx=*/2_n,
          },
          {
              MachineViewDimension{
                  stride_t{1_p},
                  MachineSpecificationDimension::INTER_NODE,
              },
              MachineViewDimension{
                  stride_t{2_p},
                  MachineSpecificationDimension::INTRA_NODE,
              },
          },
      };

      SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 0_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/2_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 1_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/4_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 0_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/2_n, /*device_idx=*/2_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 1_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/2_n, /*device_idx=*/4_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }
    }

    SUBCASE("2D case - projection on same dimension") {
      /**
       * This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
       * Both dimensions are projected on the INTRA (device) dimension, with
       * strides 1 and 2 respectively. The start of the projection defined by
       * MachineView is at MachineSpaceCoordinates (1, 0), and the machine
       * space has 2 nodes and 6 devices per node.
       *
       *  +-------+-------+-------+-------+-------+-------+
       *  | (0,0) | (1,0) |       |       | (0,1) | (1,1) |
       *  +-------+-------+-------+-------+-------+-------+
       * Where the (x,y) are the `TaskSpaceCoordinate`s, and the underlying
       * grid is the machine space.
       */

      OperatorTaskSpace task = OperatorTaskSpace{
          MinimalOrthotope{{
              2_ge2,
              2_ge2,
          }},
      };
      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/1_n,
              /*device_idx=*/0_n,
          },
          {
              MachineViewDimension{
                  stride_t{1_p},
                  MachineSpecificationDimension::INTRA_NODE,
              },
              MachineViewDimension{
                  stride_t{2_p},
                  MachineSpecificationDimension::INTRA_NODE,
              },
          },
      };

      SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 0_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/0_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 1_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/4_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 0_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/1_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 1_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/5_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }
    }

    SUBCASE("3D case") {
      /**
       * This operator has shape (2, 2, 2), and thus 2 * 2 * 2 = 8 tasks.
       * - The first dimension is projected onto the INTER (node) dimension
       * with stride 1,
       * - The second dimension is projected onto the INTRA (device) dimension
       * with stride 2,
       * - The third dimension is projected onto the INTRA (device) dimension
       * with stride 1. The start of the projection defined by MachineView is
       * at MachineSpaceCoordinates (0, 1), and the machine space has 2 nodes
       * and 8 devices per node.
       *
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+-------+-------+
       *  |       |(0,0,0)|       |(0,0,1)|       |(0,1,0)|       |(0,1,1)|
       *  +-------+-------+-------+-------+-------+-------+-------+-------+
       *  |       |(1,0,0)|       |(1,0,1)|       |(1,1,0)|       |(1,1,1)|
       *  +-------+-------+-------+-------+-------+-------+-------+-------+
       * Where the (x,y,z) are the `TaskSpaceCoordinate`s, and the underlying
       * grid is the machine space.
       */

      OperatorTaskSpace task = OperatorTaskSpace{
          MinimalOrthotope{{
              2_ge2,
              2_ge2,
              2_ge2,
          }},
      };
      MachineView mv = MachineView{
          MachineSpaceCoordinate{/*node_idx=*/0_n, /*device_idx=*/1_n},
          {MachineViewDimension{stride_t{1_p},
                                MachineSpecificationDimension::INTER_NODE},
           MachineViewDimension{stride_t{2_p},
                                MachineSpecificationDimension::INTRA_NODE},
           MachineViewDimension{stride_t{1_p},
                                MachineSpecificationDimension::INTRA_NODE}}};

      SUBCASE("Task with TaskSpaceCoordinate = (0,0,1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({0_n, 1_n, 0_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/0_n, /*device_idx=*/3_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,1,0)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 0_n, 1_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/5_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,1,1)") {
        TaskSpaceCoordinate coord = make_task_space_coordinate({1_n, 1_n, 1_n});
        MachineSpaceCoordinate correct =
            MachineSpaceCoordinate{/*node_idx=*/1_n, /*device_idx=*/7_n};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord);
        CHECK(correct == result);
      }
    }
  }
}
