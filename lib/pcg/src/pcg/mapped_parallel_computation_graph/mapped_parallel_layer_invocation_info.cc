#include "pcg/mapped_parallel_computation_graph/mapped_parallel_layer_invocation_info.h"

namespace FlexFlow {

MappedParallelLayerInvocationInfo
  mapped_parallel_layer_invocation_info_from_pcg_invocation_and_mapping(
    ParallelLayerInvocationInfo const &invocation_info,
    MappedOperatorTaskGroup const &mapping)
{
  return MappedParallelLayerInvocationInfo{
    /*incoming=*/invocation_info.incoming,
    /*layer_info=*/MappedParallelLayerInfo{
      /*guid=*/invocation_info.layer_info.guid,
      /*attrs=*/invocation_info.layer_info.attrs,
      /*mapping=*/mapping,
    },
    /*outgoing=*/invocation_info.outgoing,
  };
}


} // namespace FlexFlow
