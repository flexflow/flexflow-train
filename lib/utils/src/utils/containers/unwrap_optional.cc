#include "utils/containers/unwrap_optional.h"

namespace FlexFlow {

static_assert(std::is_same_v<unwrap_optional_t<std::optional<int>>, int>);

} // namespace FlexFlow
