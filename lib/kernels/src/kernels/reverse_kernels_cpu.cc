#include "kernels/reverse_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"
#include <vector>

namespace FlexFlow::Kernels::Reverse {

template <DataType DT>
struct CPUReverseForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  ReverseAttrs const &attrs) {
    positive_int reverse_axis_size = dim_at_idx(input.shape.dims, attrs.axis);

    for (TensorDimsCoord const &input_coord :
         get_tensor_dims_coord_set(input.shape.dims)) {
      nonnegative_int input_reverse_axis_coord =
          input_coord.ff_ordered.at(attrs.axis);

      TensorDimsCoord output_coord = input_coord;
      output_coord.ff_ordered.at(attrs.axis) =
          nonnegative_int{reverse_axis_size.int_from_positive_int() -
                          input_reverse_axis_coord.unwrap_nonnegative() - 1};

      output.at<DT>(output_coord) = input.at<DT>(input_coord);
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor,
                        ReverseAttrs const &attrs) {

  DataTypeDispatch1<CPUReverseForwardKernel>{}(
      input_accessor.shape.data_type, input_accessor, output_accessor, attrs);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad_accessor,
                         GenericTensorAccessorW &input_grad_accessor,
                         ReverseAttrs const &attrs) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(
      output_grad_accessor.shape.data_type,
      output_grad_accessor,
      input_grad_accessor,
      attrs);
}

} // namespace FlexFlow::Kernels::Reverse
