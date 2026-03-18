#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/tensor_slot_name.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "pcg/mapped_parallel_computation_graph/operator_atomic_task_shard_binding.dtg.h"
#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("adl_serializer<MappedOperatorTaskGroup>") {
    bidict<MachineSpaceCoordinate, OperatorAtomicTaskShardBinding>
        shard_bindings{
            {MachineSpaceCoordinate{0_n, 0_n, DeviceType::CPU},
             OperatorAtomicTaskShardBinding{
                 {
                     {TensorSlotName::INPUT,
                      ParallelTensorSpaceCoordinate{
                          0_n, 0_n, FFOrdered{1_n, 2_n, 3_n}}},
                 },
             }},
        };
    MappedOperatorTaskGroup deserialized{shard_bindings};
    nlohmann::json serialized = shard_bindings;

    SUBCASE("to_json") {
      nlohmann::json result = deserialized;
      nlohmann::json correct = serialized;

      CHECK(result == correct);
    }

    SUBCASE("from_json") {
      MappedOperatorTaskGroup result = serialized;
      MappedOperatorTaskGroup correct = deserialized;

      CHECK(result == correct);
    }
  }
}
