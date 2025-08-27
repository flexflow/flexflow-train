#include "compiler/cost_estimator/unstructured_communication_set.h"

namespace FlexFlow {

UnstructuredCommunicationSet
  unstructured_communication_set_from_one_to_one(
    OneToOneCommunicationSet const &one_to_one) {

  return UnstructuredCommunicationSet{
    /*raw_mapping=*/one_to_one.raw_mapping.as_unordered_map(),
  };
}

} // namespace FlexFlow
