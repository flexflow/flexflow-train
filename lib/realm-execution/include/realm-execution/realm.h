#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_H

#ifdef FLEXFLOW_USE_PREALM
#include <realm/prealm/prealm.h>
#else
#include <realm.h>
#endif

namespace FlexFlow {

#ifdef FLEXFLOW_USE_PREALM
namespace Realm = ::PRealm;
#else
namespace Realm = ::Realm;
#endif

} // namespace FlexFlow

#endif
