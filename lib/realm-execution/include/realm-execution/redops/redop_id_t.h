#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REDOPS_REALM_REDOP_ID_T_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REDOPS_REALM_REDOP_ID_T_H

#include "op-attrs/datatype.dtg.h"
#include "realm-execution/realm.h"
#include "realm-execution/redops/redop_id_t.dtg.h"

namespace FlexFlow {

/**
 * \brief Return the sum reduction operator (redop) ID for a given data type.
 */
redop_id_t get_sum_redop_id_for_data_type(DataType);

/**
   * \brief Convert a \ref FlexFlow::redop_id_t into a Realm reduction op ID.
   */
Realm::ReductionOpID get_realm_reduction_op_id_for_redop_id(redop_id_t);

} // namespace FlexFlow

#endif
