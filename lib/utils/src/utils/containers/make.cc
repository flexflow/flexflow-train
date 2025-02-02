#include "utils/containers/make.h"
#include <vector>

namespace FlexFlow {

template decltype(auto) make<std::vector<int>>();

} // namespace FlexFlow
