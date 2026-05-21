#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDOP_REGISTRY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDOP_REGISTRY_H

#include "realm-execution/realm.h"
#include "realm-execution/redops/redop_id_t.dtg.h"

namespace FlexFlow {

/**
 * \brief Registers all known reduction operators (redops).
 */
Realm::ReductionOpID get_sum_redop_id_for_data_type(DataType);

/**
   * \brief Convert a \ref FlexFlow::redop_id_t into a Realm reduction op ID.
   */
Realm::Processor::ReductionOpID
    get_realm_reduction_op_id_for_redop_id(redop_id_t);

} // namespace FlexFlow

#endif
