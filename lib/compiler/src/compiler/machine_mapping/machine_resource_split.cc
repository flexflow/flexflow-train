#include "compiler/machine_mapping/machine_resource_split.h"
#include "utils/containers/map_values.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>
  apply_resource_split(MachineResourceSplit const &split,
                       MachineComputeResourceSlice const &resources) {
    if (split.dimension == MachineSpecificationDimension::INTER_NODE) {
      ASSERT(split.offset < resources.num_nodes);

      return {
        MachineComputeResourceSlice{
          /*num_nodes=*/split.offset,
          /*num_gpus_per_node=*/resources.num_gpus_per_node,
        },
        MachineComputeResourceSlice{
          /*num_nodes=*/positive_int{
            resources.num_nodes.int_from_positive_int() - split.offset.int_from_positive_int()
          },
          /*num_gpus_per_node=*/resources.num_gpus_per_node,
        },
      };
    } else {
      ASSERT(split.dimension == MachineSpecificationDimension::INTRA_NODE);

      ASSERT(split.offset < resources.num_gpus_per_node);

      return {
        MachineComputeResourceSlice{
          /*num_nodes=*/resources.num_nodes,
          /*num_gpus_per_node=*/split.offset,
        },
        MachineComputeResourceSlice{
          /*num_nodes=*/resources.num_nodes,
          /*num_gpus_per_node=*/positive_int{
            resources.num_gpus_per_node.int_from_positive_int() - split.offset.int_from_positive_int(),
          },
        },
      };
    }
}


std::unordered_set<MachineResourceSplit>
  get_machine_resource_splits(MachineComputeResourceSlice const &resources) {
  
  std::unordered_set<MachineResourceSplit> result;

  for (positive_int i = 1_p; i < resources.num_nodes; i *= 2_p) {
    result.insert(MachineResourceSplit{
      /*offset=*/i,
      /*dimension=*/MachineSpecificationDimension::INTER_NODE,
    });
    result.insert(MachineResourceSplit{
      /*offset=*/positive_int{
        resources.num_nodes.int_from_positive_int() - i.int_from_positive_int(),
      },
      /*dimension=*/MachineSpecificationDimension::INTER_NODE,
    });
  }

  for (positive_int i = 1_p; i < resources.num_gpus_per_node; i *= 2_p) {
    result.insert(MachineResourceSplit{
      /*offset=*/i,
      /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
    });
    result.insert(MachineResourceSplit{
      /*offset=*/positive_int{
        resources.num_gpus_per_node.int_from_positive_int() - i.int_from_positive_int(),
      },
      /*dimension=*/MachineSpecificationDimension::INTRA_NODE,
    });
  }

  return result;
}

MachineSpaceCoordinate
  offset_machine_space_coordinate_by(MachineSpaceCoordinate const &coord,
                                     MachineResourceSplit const &split) {
  if (split.dimension == MachineSpecificationDimension::INTER_NODE) {
    return MachineSpaceCoordinate{
      /*node_idx=*/(coord.node_idx + split.offset).nonnegative_int_from_positive_int(),
      /*device_idx=*/coord.device_idx,
      /*device_type=*/coord.device_type,
    };
  } else {
    ASSERT(split.dimension == MachineSpecificationDimension::INTRA_NODE);

    return MachineSpaceCoordinate{
      /*node_idx=*/coord.node_idx,
      /*device_idx=*/(coord.device_idx + split.offset).nonnegative_int_from_positive_int(),
      /*device_type=*/coord.device_type,
    };
  }
}

MachineView
  offset_machine_view_by(MachineView const &machine_view,
                         MachineResourceSplit const &split) {
  return MachineView{
    /*start=*/offset_machine_space_coordinate_by(machine_view.start, split),
    /*dimensions=*/machine_view.dimensions,
  };
}

ParallelLayerGuidObliviousMachineMapping
  offset_layer_oblivious_mapping_by(ParallelLayerGuidObliviousMachineMapping const &mapping,
                                    MachineResourceSplit const &split) {

  return ParallelLayerGuidObliviousMachineMapping{
    map_values(mapping.raw_mapping,
                    [&](MachineView const &mv) {
                      return offset_machine_view_by(mv, split);
                    }),
  };
}


} // namespace FlexFlow
