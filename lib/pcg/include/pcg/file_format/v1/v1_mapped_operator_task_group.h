#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_MAPPED_OPERATOR_TASK_GROUP_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_FILE_FORMAT_V1_V1_MAPPED_OPERATOR_TASK_GROUP_H

#include "pcg/file_format/v1/v1_mapped_operator_task_group.dtg.h"
#include "pcg/mapped_parallel_computation_graph/mapped_operator_task_group.h"

namespace FlexFlow {

V1MappedOperatorTaskGroup to_v1(MappedOperatorTaskGroup const &);
MappedOperatorTaskGroup from_v1(V1MappedOperatorTaskGroup const &);

} // namespace FlexFlow

#endif
