#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_UNRESOLVED_COMMUNICATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_UNRESOLVED_COMMUNICATION_SET_H

#include <optional>
#include "compiler/cost_estimator/unresolved_communication_set.dtg.h"
#include "compiler/cost_estimator/unstructured_communication_set.dtg.h"

namespace FlexFlow {

std::optional<UnstructuredCommunicationSet>
  try_get_resolved_communication_set(UnresolvedCommunicationSet const &);

UnresolvedCommunicationSet
  unresolved_communication_set_from_communication_sets(
    std::unordered_set<UnstructuredCommunicationSet> const &);


} // namespace FlexFlow

#endif
