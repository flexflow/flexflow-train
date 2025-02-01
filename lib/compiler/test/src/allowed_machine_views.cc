#include "compiler/allowed_machine_views.h"
#include "doctest/doctest.h"
#include "utils/containers/extend.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/fmt/unordered_set.h"

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_allowed_machine_views") {

    SUBCASE("1 degree of parallelism") {
      MachineSpecification ms = MachineSpecification{
          /*num_nodes=*/1_n,
          /*num_cpus_per_node=*/5_n,
          /*num_gpus_per_node=*/5_n,
          /*inter_node_bandwidth=*/0,
          /*intra_node_bandwidth=*/0,
      };

      OperatorTaskSpace task = OperatorTaskSpace{{3_n}};

      std::unordered_set<MachineView> correct = {
          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0_n, /*device_idx=*/0_n, DeviceType::GPU},
              {MachineViewDimension{stride_t{1_n},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },

          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0_n, /*device_idx=*/1_n, DeviceType::GPU},
              {MachineViewDimension{stride_t{1_n},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },
          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0_n, /*device_idx=*/2_n, DeviceType::GPU},
              {MachineViewDimension{stride_t{1_n},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },
          MachineView{
              MachineSpaceCoordinate{
                  /*node_idx=*/0_n, /*device_idx=*/0_n, DeviceType::GPU},
              {MachineViewDimension{stride_t{2_n},
                                    MachineSpecificationDimension::INTRA_NODE}},
          },
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineSpecification ms = MachineSpecification{
          /*num_nodes=*/3_n,
          /*num_cpus_per_node=*/3_n,
          /*num_gpus_per_node=*/3_n,
          /*inter_node_bandwidth=*/0,
          /*intra_node_bandwidth=*/0,
      };
      OperatorTaskSpace task = OperatorTaskSpace{{2_n, 3_n}};

      auto make_2d_view = [&](nonnegative_int start_node_idx,
                              nonnegative_int start_device_idx,
                              nonnegative_int stride1,
                              nonnegative_int stride2,
                              MachineSpecificationDimension m1,
                              MachineSpecificationDimension m2) {
        return MachineView{
            MachineSpaceCoordinate{
                start_node_idx, start_device_idx, DeviceType::GPU},
            {MachineViewDimension{stride_t{stride1}, m1},
             MachineViewDimension{stride_t{stride2}, m2}},
        };
      };

      auto intra = MachineSpecificationDimension::INTRA_NODE;
      auto inter = MachineSpecificationDimension::INTER_NODE;
      std::unordered_set<MachineView> correct = {
          make_2d_view(
              0_n, 0_n, /*stride1=*/1_n, /*stride2=*/1_n, inter, intra),
          make_2d_view(
              1_n, 0_n, /*stride1=*/1_n, /*stride2=*/1_n, inter, intra),
          make_2d_view(
              0_n, 0_n, /*stride1=*/2_n, /*stride2=*/1_n, inter, intra),

          make_2d_view(
              0_n, 0_n, /*stride1=*/1_n, /*stride2=*/1_n, intra, inter),
          make_2d_view(
              0_n, 1_n, /*stride1=*/1_n, /*stride2=*/1_n, intra, inter),
          make_2d_view(
              0_n, 0_n, /*stride1=*/2_n, /*stride2=*/1_n, intra, inter),
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }
  }
}
