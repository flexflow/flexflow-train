#include "utils/containers/all_are_true.h"
#include <vector>

namespace FlexFlow {

using Container = std::vector<bool>;

template bool all_are_true(Container const &);

} // namespace FlexFlow
