#include "utils/half.h"
#include "utils/hash-utils.h"

namespace std {

size_t hash<half>::operator()(half h) const {
  return ::FlexFlow::get_std_hash(static_cast<float>(h));
}

} // namespace std
