#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_CG_TO_PCG_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_CG_TO_PCG_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"

namespace FlexFlow {

ParallelComputationGraph cg_to_pcg(ComputationGraph const &cg);

} // namespace FlexFlow

#endif
