#include "kernels/replicate_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow::Kernels::Replicate {

template <DataType DT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    memcpy(output.get<DT>(),
           input.get<DT>(),
           get_size_in_bytes(input.shape).unwrap_num_bytes().unwrap_nonnegative());
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
        cur_sum += output.at<DT>(TensorDimsCoord{FFOrdered{i, replica_idx}});
      }
      input.at<DT>(TensorDimsCoord{FFOrdered{i}}) = cur_sum;
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch1<CPUForwardKernel>{}(input.shape.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input,
                         size_t num_replicas) {
  positive_int num_elements = get_num_elements(input.shape.dims);
  DataTypeDispatch1<CPUBackwardKernel>{}(input.shape.data_type,
                                         output,
                                         input,
                                         num_elements,
                                         nonnegative_int{num_replicas});
}

} // namespace FlexFlow::Kernels::Replicate
