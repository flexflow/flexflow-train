#include "utils/containers/get_element_type.h"
#include <vector>

namespace FlexFlow {

static_assert(std::is_same_v<get_element_type_t<std::vector<int>>, int>);

} // namespace FlexFlow
