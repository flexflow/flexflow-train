#include "kernels/replicate_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "utils/nonnegative_int/nonnegative_range.h"

namespace FlexFlow::Kernels::Replicate {

template <DataType DT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           input.shape.num_elements().int_from_positive_int() *
               size_of_datatype(DT).int_from_positive_int());
  }
};

template <DataType DT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output,
                  GenericTensorAccessorW const &input,
                  positive_int num_elements,
                  nonnegative_int num_replicas) {
    using T = real_type_t<DT>;

    for (nonnegative_int i :
         nonnegative_range(num_elements.nonnegative_int_from_positive_int())) {
      T cur_sum = 0;
      for (nonnegative_int replica_idx : nonnegative_range(num_replicas)) {
        cur_sum += output.at<DT>(LegionOrdered{replica_idx, i});
      }
      input.at<DT>(LegionOrdered{i}) = cur_sum;
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(input.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input,
                         size_t num_replicas) {
  positive_int num_elements = input.shape.num_elements();
  DataTypeDispatch1<CPUBackwardKernel>{}(input.data_type,
                                         output,
                                         input,
                                         num_elements,
                                         nonnegative_int{num_replicas});
}

} // namespace FlexFlow::Kernels::Replicate
