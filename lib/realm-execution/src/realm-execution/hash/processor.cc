#include "realm-execution/hash/processor.h"
#include <utility>

namespace std {

size_t hash<::FlexFlow::Realm::Processor>::operator()(
    ::FlexFlow::Realm::Processor const &p) const {
  return hash<::FlexFlow::Realm::Processor::id_t>{}(p.id);
}

} // namespace std
