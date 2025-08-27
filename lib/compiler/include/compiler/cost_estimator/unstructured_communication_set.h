#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_UNSTRUCTURED_COMMUNICATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_COST_ESTIMATOR_UNSTRUCTURED_COMMUNICATION_SET_H

#include "compiler/cost_estimator/one_to_one_communication_set.dtg.h"
#include "compiler/cost_estimator/unstructured_communication_set.dtg.h"

namespace FlexFlow {

UnstructuredCommunicationSet
  unstructured_communication_set_from_one_to_one(
    OneToOneCommunicationSet const &one_to_one);

} // namespace FlexFlow

#endif
