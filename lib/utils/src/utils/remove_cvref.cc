#include "utils/remove_cvref.h"

namespace FlexFlow {

static_assert(std::is_same_v<remove_cvref_t<int const &>, int>);

} // namespace FlexFlow
