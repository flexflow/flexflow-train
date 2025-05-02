#include "kernels/datatype_dispatch.h"
#include "kernels/reverse_kernels_cpu.h"
#include <vector>

namespace FlexFlow::Kernels::Reverse {

template <DataType DT>
struct CPUReverseForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  ReverseAttrs const &attrs) {
    nonnegative_int reverse_axis_size = input.shape.at(attrs.axis);

    for (ArrayCoord const &input_coord : get_array_coord_set(input.shape)) {
      nonnegative_int input_reverse_axis_coord =
          input_coord.ff_ordered.at(attrs.axis);

      ArrayCoord output_coord = input_coord;
      output_coord.ff_ordered.at(attrs.axis) =
          nonnegative_int{reverse_axis_size.unwrap_nonnegative() -
                          input_reverse_axis_coord.unwrap_nonnegative() - 1};

      output.at<DT>(output_coord.ff_ordered) =
          input.at<DT>(input_coord.ff_ordered);
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor,
                        ReverseAttrs const &attrs) {

  DataTypeDispatch1<CPUReverseForwardKernel>{}(
      input_accessor.data_type, input_accessor, output_accessor, attrs);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_grad_accessor,
                         GenericTensorAccessorW &input_grad_accessor,
                         ReverseAttrs const &attrs) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(output_grad_accessor.data_type,
                                               output_grad_accessor,
                                               input_grad_accessor,
                                               attrs);
}

} // namespace FlexFlow::Kernels::Reverse
