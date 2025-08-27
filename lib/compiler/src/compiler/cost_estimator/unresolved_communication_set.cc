#include "compiler/cost_estimator/unresolved_communication_set.h"
#include "utils/containers/get_only.h"

namespace FlexFlow {


std::optional<UnstructuredCommunicationSet>
  try_get_resolved_communication_set(UnresolvedCommunicationSet const &unresolved) {

    std::unordered_map<MachineSpaceCoordinate, MachineSpaceCoordinate> resolved;

  for (MachineSpaceCoordinate const &dst : unresolved.raw_mapping.right_values()) {
    std::unordered_set<MachineSpaceCoordinate> src_options = unresolved.raw_mapping.at_r(dst);

    ASSERT(!src_options.empty());

    if (src_options.size() > 1) {
      return std::nullopt;
    }

    resolved.insert({get_only(src_options), dst});
  }

  return UnstructuredCommunicationSet{
    /*raw_mapping=*/resolved,
  };
}

UnresolvedCommunicationSet
  unresolved_communication_set_from_communication_sets(
    std::unordered_set<UnstructuredCommunicationSet> const &communication_sets) {

  ManyToOne<MachineSpaceCoordinate, MachineSpaceCoordinate> result;

  for (auto const &communication_set : communication_sets) {
    for (auto const &communication : communication_set.raw_mapping) {
      result.insert(communication);
    }
  }

  return UnresolvedCommunicationSet{
    /*raw_mapping=*/result,
  };
}



} // namespace FlexFlow
