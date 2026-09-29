#include "compiler/search_result.h"
#include "compiler/machine_mapping/machine_view.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/pcg_from_computation_graph.h"
#include "utils/containers/generate_map.h"

namespace FlexFlow {

MappedParallelComputationGraph
    get_mapped_pcg_from_search_result(SearchResult const &search_result) {
  return mapped_pcg_from_pcg_and_mapping(search_result.pcg,
                                         search_result.machine_mapping);
}

std::string format_as(SearchResult const &r) {
  return fmt::format("<SearchResult\npcg={}\nmachine_mapping={}>",
                     pcg_as_dot(r.pcg),
                     r.machine_mapping);
}

static std::pair<parallel_layer_guid_t, MappedOperatorTaskGroup>
    single_device_mapping_from_pcg_invocation_info(
        ParallelLayerInvocationInfo const &info) {
  // Everything maps to zero
  ParallelTensorSpaceCoordinate tensor_coord_zero{
      0_n, 0_n, FFOrdered<nonnegative_int>{0_n}};
  MachineSpaceCoordinate machine_coord_zero{0_n, 0_n};
  OperatorAtomicTaskShardBinding shard_binding{binary_merge_disjoint_maps(
      map_values(info.incoming,
                 [&](ParallelTensorInfo const &) { return tensor_coord_zero; }),
      map_values(info.outgoing, [&](ParallelTensorInfo const &) {
        return tensor_coord_zero;
      }))};
  return std::pair<parallel_layer_guid_t, MappedOperatorTaskGroup>{
      info.layer_info.guid,
      MappedOperatorTaskGroup{{{machine_coord_zero, shard_binding}}},
  };
}

SearchResult trivial_search_result_for_cg(ComputationGraph const &cg,
                                          MachineSpaceCoordinate const &mc) {
  ParallelComputationGraph pcg = pcg_from_computation_graph(cg);
  return SearchResult{
      /*pcg=*/pcg,
      /*machine_mapping=*/
      MachineMapping{
          generate_map(pcg_get_parallel_layers(pcg),
                       [&](parallel_layer_guid_t l) -> MachineView {
                         return make_single_device_machine_view(mc);
                       }),
      },
  };
}

std::ostream &operator<<(std::ostream &s, SearchResult const &r) {
  return (s << fmt::to_string(r));
}

} // namespace FlexFlow
