#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_HASH_PROCESSOR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_HASH_PROCESSOR_H

#include "realm-execution/realm.h"
#include <utility>

namespace std {

template <>
struct hash<::FlexFlow::Realm::Processor> {
  size_t operator()(::FlexFlow::Realm::Processor const &p) const;
};

} // namespace std

#endif
