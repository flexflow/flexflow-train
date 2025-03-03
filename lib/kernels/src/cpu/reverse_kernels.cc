#include "kernels/datatype_dispatch.h"
#include "kernels/reverse_kernels_cpu.h"
#include <vector>

namespace FlexFlow::Kernels::Reverse {

template <DataType DT>
struct CPUReverseForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW &output,
                  int num_out_blks,
                  int reverse_dim_size,
                  int in_blk_size) {
    for (int blk_idx = 0; blk_idx < num_out_blks; blk_idx++) {
      for (int rev_idx = 0; rev_idx < reverse_dim_size; rev_idx++) {
        for (int inner_idx = 0; inner_idx < in_blk_size; inner_idx++) {
          output.at<DT>({inner_idx, rev_idx, blk_idx}) = input.at<DT>(
              {inner_idx, reverse_dim_size - 1 - rev_idx, blk_idx});
        }
      }
    }
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input_accessor,
                        GenericTensorAccessorW &output_accessor,
                        int num_out_blks,
                        int reverse_dim_size,
                        int in_blk_size) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(input_accessor.data_type,
                                               input_accessor,
                                               output_accessor,
                                               num_out_blks,
                                               reverse_dim_size,
                                               in_blk_size);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output_accessor,
                         GenericTensorAccessorW &input_accessor,
                         int num_out_blks,
                         int reverse_dim_size,
                         int in_blk_size) {
  DataTypeDispatch1<CPUReverseForwardKernel>{}(output_accessor.data_type,
                                               output_accessor,
                                               input_accessor,
                                               num_out_blks,
                                               reverse_dim_size,
                                               in_blk_size);
}

} // namespace FlexFlow::Kernels::Reverse
