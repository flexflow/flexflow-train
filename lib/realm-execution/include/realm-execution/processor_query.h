#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PROCESSOR_QUERY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_PROCESSOR_QUERY_H

#include "realm-execution/realm.h"

namespace FlexFlow {

std::set<Realm::Processor> processor_set_from_query(Realm::Machine::ProcessorQuery const &);

} // namespace FlexFlow

#endif
