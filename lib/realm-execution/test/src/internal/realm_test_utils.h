#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_TEST_SRC_INTERNAL_REALM_TEST_UTILS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_TEST_SRC_INTERNAL_REALM_TEST_UTILS_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/positive_int/positive_int.h"
#include <vector>

namespace FlexFlow {

std::vector<char *> make_fake_realm_args(positive_int num_cpus,
                                         nonnegative_int num_gpus);

} // namespace FlexFlow

#endif
