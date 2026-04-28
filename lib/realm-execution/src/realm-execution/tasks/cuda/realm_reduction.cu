// realm_reduction_cuda.cu
#include "realm-execution/tasks/realm_reduction.h"
#include <realm.h>
#include <realm/cuda/cuda_redop.h>
#include <realm/redop.h>

namespace FlexFlow {

void register_reductions() {
  ::Realm::Runtime rt = ::Realm::Runtime::get_runtime();

  // register SumReductionFloat with CUDA kernels
  {
    ::Realm::ReductionOpUntyped *redop =
        ::Realm::ReductionOpUntyped::create_reduction_op<SumReductionFloat>();
    ::Realm::Cuda::add_cuda_redop_kernels<SumReductionFloat>(redop);
    bool ok = rt.register_reduction(REDOP_SUM_FLOAT, redop);
    assert(ok && "Failed to register SumReductionFloat");
  }

  // register SumReductionDouble with CUDA kernels
  {
    ::Realm::ReductionOpUntyped *redop =
        ::Realm::ReductionOpUntyped::create_reduction_op<SumReductionDouble>();
    ::Realm::Cuda::add_cuda_redop_kernels<SumReductionDouble>(redop);
    bool ok = rt.register_reduction(REDOP_SUM_DOUBLE, redop);
    assert(ok && "Failed to register SumReductionDouble");
  }
}

} // namespace FlexFlow
