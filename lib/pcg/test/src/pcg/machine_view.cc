#include "pcg/machine_view.h"
#include "pcg/gpu_id_t.dtg.h"
#include "test/utils/doctest/fmt/optional.h"
#include "utils/containers/transform.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/vector.h"
#include <doctest/doctest.h>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("MachineView - utility functions") {
    MachineView mv = MachineView{
        MachineSpaceCoordinate{
            /*node_idx=*/0_n, /*device_idx=*/0_n, DeviceType::GPU},
        {MachineViewDimension{stride_t{2_n},
                              MachineSpecificationDimension::INTER_NODE},
         MachineViewDimension{stride_t{2_n},
                              MachineSpecificationDimension::INTER_NODE}}};

    SUBCASE("num_dims") {
      CHECK(num_dims(mv) == 2);
    }
    SUBCASE("get_device_type") {
      CHECK(get_device_type(mv) == DeviceType::GPU);
    }
  }

  TEST_CASE("get_machine_space_coordinate") {
    SUBCASE("1D case") {

      // This operator has shape (3,), and thus 3 tasks.
      // The (only) dimension is projected on the INTER (device) dimension with
      // a stride of 2. The start of the projection defined by MachineView
      // starts at MachineSpaceCoordinate (0,1), and the machine space has 1
      // node and 6 devices per node.

      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+
       *  |       | (0,)  |       | (1,)  |       | (2,)  |
       *  +-------+-------+-------+-------+-------+-------+
       * Where the (x,) are the `TaskSpaceCoordinate`s, and the underlying grid
       * is the machine space.
       */
      OperatorTaskSpace task = OperatorTaskSpace{{3_n}};
      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/0_n, /*device_idx=*/1_n, DeviceType::GPU},
          {MachineViewDimension{stride_t{2_n},
                                MachineSpecificationDimension::INTRA_NODE}}};
      MachineSpecification ms =
          MachineSpecification{/*num_nodes=*/1_n,
                               /*num_cpus_per_node=*/6_n,
                               /*num_gpus_per_node=*/6_n,
                               /*inter_node_bandwidth=*/0,
                               /*intra_node_bandwidth=*/0};

      SUBCASE("Task with TaskSpaceCoordinate = (0,)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0_n}};
        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0_n, /*device_idx=*/1_n, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms).value();
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (1,)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n}};
        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0_n, /*device_idx=*/3_n, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms).value();
        CHECK(correct == result);
      }

      SUBCASE("Task with TaskSpaceCoordinate = (2,)") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{2_n}};
        MachineSpaceCoordinate correct = MachineSpaceCoordinate{
            /*node_idx=*/0_n, /*device_idx=*/5_n, DeviceType::GPU};
        MachineSpaceCoordinate result =
            get_machine_space_coordinate(task, mv, coord, ms).value();
        CHECK(correct == result);
      }

      SUBCASE("TaskSpaceCoordinate is out of bounds") {
        TaskSpaceCoordinate coord = TaskSpaceCoordinate{{4_n}};
        std::optional<MachineSpaceCoordinate> result =
            get_machine_space_coordinate(task, mv, coord, ms);
        std::optional<MachineSpaceCoordinate> correct = std::nullopt;
        CHECK(result == correct);
      }

      SUBCASE("2D case - projection on different dimensions") {
        // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
        // The first dimension is projected onto the INTER (node) dimension with
        // stride 1, while the second dimension is projected onto the INTRA
        // (device) dimension with stride 2. The start of the projection defined
        // by MachineView is at MachineSpaceCoordinates (1, 2), and the machine
        // space has 3 nodes and 5 devices per node.

        /**
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

        OperatorTaskSpace task = OperatorTaskSpace{{2_n, 2_n}};
        MachineView mv = MachineView{
            MachineSpaceCoordinate{
                /*node_idx=*/1_n, /*device_idx=*/2_n, DeviceType::GPU},
            {MachineViewDimension{stride_t{1_n},
                                  MachineSpecificationDimension::INTER_NODE},
             MachineViewDimension{stride_t{2_n},
                                  MachineSpecificationDimension::INTRA_NODE}}};
        MachineSpecification ms =
            MachineSpecification{/*num_nodes=*/3_n,
                                 /*num_cpus_per_node=*/5_n,
                                 /*num_gpus_per_node=*/5_n,
                                 /*inter_node_bandwidth=*/0,
                                 /*intra_node_bandwidth=*/0};

        SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0_n, 0_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/2_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0_n, 1_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/4_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n, 0_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/2_n, /*device_idx=*/2_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n, 1_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/2_n, /*device_idx=*/4_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }
      }

      SUBCASE("2D case - projection on same dimension") {
        // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
        // Both dimensions are projected on the INTRA (device) dimension, with
        // strides 1 and 2 respectively. The start of the projection defined by
        // MachineView is at MachineSpaceCoordinates (1, 0), and the machine
        // space has 2 nodes and 6 devices per node.

        /**
         *  +-------+-------+-------+-------+-------+-------+
         *  | (0,0) | (1,0) |       |       | (0,1) | (1,1) |
         *  +-------+-------+-------+-------+-------+-------+
         * Where the (x,y) are the `TaskSpaceCoordinate`s, and the underlying
         * grid is the machine space.
         */

        OperatorTaskSpace task = OperatorTaskSpace{{2_n, 2_n}};
        MachineView mv = MachineView{
            MachineSpaceCoordinate{
                /*node_idx=*/1_n, /*device_idx=*/0_n, DeviceType::GPU},
            {MachineViewDimension{stride_t{1_n},
                                  MachineSpecificationDimension::INTRA_NODE},
             MachineViewDimension{stride_t{2_n},
                                  MachineSpecificationDimension::INTRA_NODE}}};
        MachineSpecification ms =
            MachineSpecification{/*num_nodes=*/2_n,
                                 /*num_cpus_per_node=*/6_n,
                                 /*num_gpus_per_node=*/6_n,
                                 /*inter_node_bandwidth=*/0,
                                 /*intra_node_bandwidth=*/0};

        SUBCASE("Task with TaskSpaceCoordinate = (0,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0_n, 0_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/0_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0_n, 1_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/4_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n, 0_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/1_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n, 1_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/5_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }
      }

      SUBCASE("3D case") {
        // This operator has shape (2, 2, 2), and thus 2 * 2 * 2 = 8 tasks.
        // - The first dimension is projected onto the INTER (node) dimension
        // with stride 1,
        // - The second dimension is projected onto the INTRA (device) dimension
        // with stride 2,
        // - The third dimension is projected onto the INTRA (device) dimension
        // with stride 1. The start of the projection defined by MachineView is
        // at MachineSpaceCoordinates (0, 1), and the machine space has 2 nodes
        // and 8 devices per node.

        /**
         * The tasks will thus be distributed like this:
         *  +-------+-------+-------+-------+-------+-------+-------+-------+
         *  |       |(0,0,0)|       |(0,0,1)|       |(0,1,0)|       |(0,1,1)|
         *  +-------+-------+-------+-------+-------+-------+-------+-------+
         *  |       |(1,0,0)|       |(1,0,1)|       |(1,1,0)|       |(1,1,1)|
         *  +-------+-------+-------+-------+-------+-------+-------+-------+
         * Where the (x,y,z) are the `TaskSpaceCoordinate`s, and the underlying
         * grid is the machine space.
         */

        OperatorTaskSpace task = OperatorTaskSpace{{2_n, 2_n, 2_n}};
        MachineView mv = MachineView{
            MachineSpaceCoordinate{
                /*node_idx=*/0_n, /*device_idx=*/1_n, DeviceType::GPU},
            {MachineViewDimension{stride_t{1_n},
                                  MachineSpecificationDimension::INTER_NODE},
             MachineViewDimension{stride_t{2_n},
                                  MachineSpecificationDimension::INTRA_NODE},
             MachineViewDimension{stride_t{1_n},
                                  MachineSpecificationDimension::INTRA_NODE}}};
        MachineSpecification ms =
            MachineSpecification{/*num_nodes=*/2_n,
                                 /*num_cpus_per_node=*/8_n,
                                 /*num_gpus_per_node=*/8_n,
                                 /*inter_node_bandwidth=*/0,
                                 /*intra_node_bandwidth=*/0};

        SUBCASE("Task with TaskSpaceCoordinate = (0,0,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{0_n, 1_n, 0_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/0_n, /*device_idx=*/3_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1,0)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n, 0_n, 1_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/5_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }

        SUBCASE("Task with TaskSpaceCoordinate = (1,1,1)") {
          TaskSpaceCoordinate coord = TaskSpaceCoordinate{{1_n, 1_n, 1_n}};
          MachineSpaceCoordinate correct = MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/7_n, DeviceType::GPU};
          MachineSpaceCoordinate result =
              get_machine_space_coordinate(task, mv, coord, ms).value();
          CHECK(correct == result);
        }
      }
    }
  }

  TEST_CASE("get_device_ids") {

    SUBCASE("1D machine view") {

      // This operator has shape (3,), and thus 3 tasks.
      // The (only) dimension is projected onto the INTRA (device) dimension
      // with a stride of 2. The start of the projection defined by MachineView
      // is at MachineSpaceCoordinate (0, 1), and the machine space has 1 node
      // and 6 devices per node.

      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+-------+
       *  |   0   | ((1)) |   2   | ((3)) |   4   | ((5)) |
       *  +-------+-------+-------+-------+-------+-------+
       * Where the integers are the device ids and ((x)) are the devices we
       * select
       */
      MachineSpecification ms =
          MachineSpecification{/*num_nodes=*/1_n,
                               /*num_cpus_per_node=*/6_n,
                               /*num_gpus_per_node=*/6_n,
                               /*inter_node_bandwidth=*/0,
                               /*intra_node_bandwidth=*/0};

      OperatorTaskSpace task = OperatorTaskSpace{{3_n}};
      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/0_n, /*device_idx=*/1_n, DeviceType::GPU},
          {MachineViewDimension{stride_t{2_n},
                                MachineSpecificationDimension::INTRA_NODE}}};

      std::unordered_set<device_id_t> correct = {
          device_id_t{gpu_id_t{1_n}},
          device_id_t{gpu_id_t{3_n}},
          device_id_t{gpu_id_t{5_n}},
      };
      std::unordered_set<device_id_t> result = get_device_ids(task, mv, ms);
      CHECK(result == correct);
    }

    SUBCASE("2D machine view") {
      // This operator has shape (2, 2), and thus 2 * 2 = 4 tasks.
      // - The first dimension is projected onto the INTER (node) dimension with
      // stride 1,
      // - The second dimension is projected onto the INTRA (device) dimension
      // with stride 2. The start of the projection defined by MachineView is at
      // MachineSpaceCoordinate (1, 2), and the machine space has 3 nodes and 5
      // devices per node.

      /**
       * The tasks will thus be distributed like this:
       *  +-------+-------+-------+-------+-------+
       *  |   0   |   1   |   2   |   3   |   4   |
       *  +-------+-------+-------+-------+-------+
       *  |   5   |   6   | ((7)) |   8   | ((9)) |
       *  +-------+-------+-------+-------+-------+
       *  |   10  |   11  | ((12))|  13   | ((14))|
       *  +-------+-------+-------+-------+-------+
       * Where the integers are the device ids and ((x)) are the devices we
       * select
       */

      MachineSpecification ms =
          MachineSpecification{/*num_nodes=*/3_n,
                               /*num_cpus_per_node=*/5_n,
                               /*num_gpus_per_node=*/5_n,
                               /*inter_node_bandwidth=*/0,
                               /*intra_node_bandwidth=*/0};

      OperatorTaskSpace task = OperatorTaskSpace{{2_n, 2_n}};
      MachineView mv = MachineView{
          MachineSpaceCoordinate{
              /*node_idx=*/1_n, /*device_idx=*/2_n, DeviceType::GPU},
          {MachineViewDimension{stride_t{1_n},
                                MachineSpecificationDimension::INTER_NODE},
           MachineViewDimension{stride_t{2_n},
                                MachineSpecificationDimension::INTRA_NODE}}};

      std::unordered_set<device_id_t> correct = {
          device_id_t{gpu_id_t{7_n}},
          device_id_t{gpu_id_t{9_n}},
          device_id_t{gpu_id_t{12_n}},
          device_id_t{gpu_id_t{14_n}},
      };
      std::unordered_set<device_id_t> result = get_device_ids(task, mv, ms);
      CHECK(result == correct);
    }
  }
}
