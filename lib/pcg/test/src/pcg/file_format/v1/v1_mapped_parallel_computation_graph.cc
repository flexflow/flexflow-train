#include "pcg/file_format/v1/v1_mapped_parallel_computation_graph.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/file_format/v1/v1_mapped_operator_task_group.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "pcg/mapped_parallel_computation_graph/mapped_parallel_computation_graph.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_layer_added_result.dtg.h"
#include "utils/bidict/bidict.h"
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("V1MappedParallelComputationGraph") {
    ParallelComputationGraph pcg = empty_parallel_computation_graph();

    TensorShape input_shape = TensorShape{
        TensorDims{
            FFOrdered{
                12_p,
                16_p,
            },
        },
        DataType::FLOAT,
    };

    ParallelLayerAddedResult result = pcg_add_input_layer(pcg, input_shape);
    parallel_layer_guid_t layer = result.parallel_layer;

    MachineSpaceCoordinate coord = MachineSpaceCoordinate{
        /*node_idx=*/0_n,
        /*device_idx=*/0_n,
        /*device_type=*/DeviceType::GPU,
    };

    OperatorAtomicTaskShardBinding binding = OperatorAtomicTaskShardBinding{
        /*tensor_coords=*/{
            {
                TensorSlotName::OUTPUT,
                ParallelTensorSpaceCoordinate{
                    /*sum_component=*/0_n,
                    /*discard_copy_component=*/0_n,
                    /*shard_components=*/FFOrdered<nonnegative_int>{0_n, 0_n},
                },
            },
        },
    };

    MappedOperatorTaskGroup task_group = MappedOperatorTaskGroup{
        bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>{
            {coord, binding},
        },
    };

    MappedParallelComputationGraph mpcg = MappedParallelComputationGraph{
        /*pcg=*/pcg,
        /*mapped_tasks=*/{{layer, task_group}},
    };

    V1MappedParallelComputationGraph v1_mpcg = to_v1(mpcg);

    SUBCASE("serializes to JSON") {
      nlohmann::json j = v1_mpcg;
    }

    SUBCASE("MappedOperatorTaskGroup round-trips via from_v1") {
      MappedOperatorTaskGroup result = from_v1(to_v1(task_group));
      CHECK(result == task_group);
    }

    SUBCASE("MappedParallelComputationGraph round-trips via from_v1") {
      MappedParallelComputationGraph result = from_v1(v1_mpcg);
      CHECK(pcgs_are_isomorphic(mpcg.pcg, result.pcg));
    }
  }
}
