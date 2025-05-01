#include "kernels/datatype_dispatch.h"
#include "kernels/replicate_kernels_cpu.h"

namespace FlexFlow::Kernels::Replicate {

template <DataType DT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           input.shape.num_elements().unwrap_nonnegative() *
               size_of_datatype(DT).unwrap_nonnegative());
  }
};

template <DataType DT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output,
                  GenericTensorAccessorW &input,
                  nonnegative_int num_elements,
                  nonnegative_int num_replicas) {
    using T = real_type_t<DT>;

    for (nonnegative_int i : nonnegative_range(num_elements)) {
      T cur_sum = 0;
      for (nonnegative_int replica_idx : nonnegative_range(num_replicas)) {
        cur_sum += output.at<DT>(LegionOrdered{i, replica_idx});
      }
      input.at<DT>(LegionOrdered{i}) = cur_sum;
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(input.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW &input,
                         size_t num_replicas) {
  nonnegative_int num_elements = input.shape.num_elements();
  DataTypeDispatch1<CPUBackwardKernel>{}(input.data_type,
                                         output,
                                         input,
                                         num_elements,
                                         nonnegative_int{num_replicas});
}

} // namespace FlexFlow::Kernels::Replicate
