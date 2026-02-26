#include "compiler/machine_mapping/allowed_machine_views.h"
#include "doctest/doctest.h"
#include "utils/containers/extend.h"
#include "utils/containers/range.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/zip.h"
#include "utils/fmt/unordered_set.h"
#include <libassert/assert.hpp>

using namespace FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {

  TEST_CASE("get_allowed_machine_views") {

    auto make_machine_view =
        [&](nonnegative_int start_node_idx,
            nonnegative_int start_device_idx,
            std::optional<positive_int> stride_1 = std::nullopt,
            std::optional<MachineSpecificationDimension> m1 = std::nullopt,
            std::optional<positive_int> stride_2 = std::nullopt,
            std::optional<MachineSpecificationDimension> m2 = std::nullopt) {
          std::vector<MachineViewDimension> strides;

          if (stride_1.has_value()) {
            ASSERT(m1.has_value());
            strides.push_back(
                MachineViewDimension{stride_t{stride_1.value()}, m1.value()});
          }

          if (stride_2.has_value()) {
            ASSERT(stride_1.has_value());
            ASSERT(m2.has_value());
            strides.push_back(
                MachineViewDimension{stride_t{stride_2.value()}, m2.value()});
          }

          return MachineView{
              MachineSpaceCoordinate{
                  start_node_idx,
                  start_device_idx,
                  DeviceType::GPU,
              },
              strides,
          };
        };

    auto intra = MachineSpecificationDimension::INTRA_NODE;
    auto inter = MachineSpecificationDimension::INTER_NODE;

    SUBCASE("1 degree of parallelism") {
      MachineComputeResourceSlice ms = MachineComputeResourceSlice{
          /*num_nodes=*/1_p,
          /*num_gpus_per_node=*/5_p,
      };

      OperatorTaskSpace task = OperatorTaskSpace{MinimalOrthotope{{3_ge2}}};

      std::unordered_set<MachineView> correct = {
          make_machine_view(0_n, 0_n, 1_p, intra),
          make_machine_view(0_n, 1_n, 1_p, intra),
          make_machine_view(0_n, 2_n, 1_p, intra),
          make_machine_view(0_n, 0_n, 2_p, intra),
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }

    SUBCASE("2 degrees of parallelism") {

      MachineComputeResourceSlice ms = MachineComputeResourceSlice{
          /*num_nodes=*/3_p,
          /*num_gpus_per_node=*/3_p,
      };
      OperatorTaskSpace task =
          OperatorTaskSpace{MinimalOrthotope{{2_ge2, 3_ge2}}};

      std::unordered_set<MachineView> correct = {
          make_machine_view(
              0_n, 0_n, /*stride_1=*/1_p, inter, /*stride_2=*/1_p, intra),
          make_machine_view(
              1_n, 0_n, /*stride_1=*/1_p, inter, /*stride_2=*/1_p, intra),
          make_machine_view(
              0_n, 0_n, /*stride_1=*/2_p, inter, /*stride_2=*/1_p, intra),

          make_machine_view(
              0_n, 0_n, /*stride_1=*/1_p, intra, /*stride_2=*/1_p, inter),
          make_machine_view(
              0_n, 1_n, /*stride_1=*/1_p, intra, /*stride_2=*/1_p, inter),
          make_machine_view(
              0_n, 0_n, /*stride_1=*/2_p, intra, /*stride_2=*/1_p, inter),
      };

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(ms, task, DeviceType::GPU);

      CHECK(correct == result);
    }

    SUBCASE("2D operator task space, dimensions (1,1)") {
      MachineComputeResourceSlice full_machine_spec =
          MachineComputeResourceSlice{
              /*num_nodes=*/2_p,
              /*num_gpus_per_node=*/1_p,
          };
      OperatorTaskSpace task = OperatorTaskSpace{MinimalOrthotope{{}}};

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(full_machine_spec, task, DeviceType::GPU);

      std::unordered_set<MachineView> correct = {
          make_machine_view(0_n, 0_n),
          make_machine_view(1_n, 0_n),
      };

      CHECK(correct == result);
    }

    SUBCASE("2D operator task space, dimensions (2,1)") {
      MachineComputeResourceSlice full_machine_spec =
          MachineComputeResourceSlice{
              /*num_nodes=*/2_p,
              /*num_gpus_per_node=*/2_p,
          };
      OperatorTaskSpace task = OperatorTaskSpace{MinimalOrthotope{{2_ge2}}};

      std::unordered_set<MachineView> result =
          get_allowed_machine_views(full_machine_spec, task, DeviceType::GPU);

      std::unordered_set<MachineView> correct = {
          make_machine_view(0_n, 0_n, /*stride_1=*/1_p, intra),
          make_machine_view(0_n, 0_n, /*stride_1=*/1_p, inter),
          make_machine_view(1_n, 0_n, /*stride_1=*/1_p, intra),
          make_machine_view(0_n, 1_n, /*stride_1=*/1_p, inter)};

      CHECK(correct == result);
    }
  }
}
