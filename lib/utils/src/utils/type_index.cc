#include "utils/type_index.h"

namespace FlexFlow {

template std::type_index get_type_index_for_type<int>();

template bool matches<int>(std::type_index);

} // namespace FlexFlow
