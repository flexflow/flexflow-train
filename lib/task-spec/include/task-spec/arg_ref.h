#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_ARG_REF_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_ARG_REF_H

namespace FlexFlow {

template <typename LABEL_TYPE, typename T>
struct ArgRef {
  LABEL_TYPE ref_type;
};

} // namespace FlexFlow

#endif
