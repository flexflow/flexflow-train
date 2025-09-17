#include "compiler/machine_mapping/machine_resource_split.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "test/utils/doctest/fmt/pair.h"
#include "test/utils/doctest/fmt/unordered_set.h"
#include "utils/hash/pair.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("get_machine_resource_splits") {
    SUBCASE("returns no splits if no splits are possible") {
      MachineComputeResourceSlice input = MachineComputeResourceSlice{
        /*num_nodes=*/1_p,
        /*num_gpus_per_node=*/1_p,
      };

      std::unordered_set<MachineResourceSplit>
          result = get_machine_resource_splits(input);
      std::unordered_set<MachineResourceSplit>
          correct = {};

      CHECK(result == correct);
    }

    SUBCASE(
        "returns splits in gpu and node dimensions") {
      MachineComputeResourceSlice input = MachineComputeResourceSlice{
        /*num_nodes=*/2_p,
        /*num_gpus_per_node=*/2_p,
      };

      std::unordered_set<MachineResourceSplit>
          result = get_machine_resource_splits(input);
      
      std::unordered_set<MachineResourceSplit> 
          correct = {
        MachineResourceSplit{
          /*offset=*/1_p,
          /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
        },
        MachineResourceSplit{
          /*offset=*/1_p,
          /*dimension=*/MachineSpecificationDimension::INTER_NODE,
        }
      };

      CHECK(result == correct);
    }

    SUBCASE("returns splits in node dimension in powers of two") {
      MachineComputeResourceSlice input = MachineComputeResourceSlice{
        /*num_nodes=*/8_p,
        /*num_gpus_per_node=*/1_p,
      };

      std::unordered_set<MachineResourceSplit>
          result = get_machine_resource_splits(input);

      std::unordered_set<MachineResourceSplit>
          correct = {
        MachineResourceSplit{
          /*offset=*/1_p,
          /*dimension=*/MachineSpecificationDimension::INTER_NODE,
        },
        MachineResourceSplit{
          /*offset=*/2_p,
          /*dimension=*/MachineSpecificationDimension::INTER_NODE,
        },
        MachineResourceSplit{
          /*offset=*/4_p,
          /*dimension=*/MachineSpecificationDimension::INTER_NODE,
        },
        MachineResourceSplit{
          /*offset=*/6_p,
          /*dimension=*/MachineSpecificationDimension::INTER_NODE,
        },
        MachineResourceSplit{
          /*offset=*/7_p,
          /*dimension=*/MachineSpecificationDimension::INTER_NODE,
        },
      };

      CHECK(result == correct);
    }

    SUBCASE("returns splits in gpu dimension in powers of two") {
      MachineComputeResourceSlice input = MachineComputeResourceSlice{
        /*num_nodes=*/1_p,
        /*num_gpus_per_node=*/8_p,
      };

      std::unordered_set<MachineResourceSplit>
          result = get_machine_resource_splits(input);

      std::unordered_set<MachineResourceSplit>
          correct = {
        MachineResourceSplit{
          /*offset=*/1_p,
          /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
        },
        MachineResourceSplit{
          /*offset=*/2_p,
          /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
        },
        MachineResourceSplit{
          /*offset=*/4_p,
          /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
        },
        MachineResourceSplit{
          /*offset=*/6_p,
          /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
        },
        MachineResourceSplit{
          /*offset=*/7_p,
          /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
        },
      };

      CHECK(result == correct);
    }
  }
}
