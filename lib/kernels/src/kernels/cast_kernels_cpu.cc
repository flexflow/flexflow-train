#include "kernels/cast_kernels_cpu.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow::Kernels::Cast {

template <typename IDT, typename ODT>
void cpu_cast_forward(IDT const *input, ODT *output, size_t volume) {
  for (size_t i = 0; i < volume; ++i) {
    output[i] = static_cast<ODT>(input[i]);
  }
}

template <typename IDT, typename ODT>
void cpu_cast_backward(IDT const *input, ODT *output, size_t volume, ODT beta) {
  for (size_t i = 0; i < volume; i++) {
    output[i] = static_cast<ODT>(input[i]) + beta * output[i];
  }
}

template <DataType IDT, DataType ODT>
struct CPUForwardKernel {
  void operator()(GenericTensorAccessorR const &input,
                  GenericTensorAccessorW const &output) {
    size_t volume = get_num_elements(input.shape.dims).int_from_positive_int();
    cpu_cast_forward(input.get<IDT>(), output.get<ODT>(), volume);
  }
};

template <DataType IDT, DataType ODT>
struct CPUBackwardKernel {
  void operator()(GenericTensorAccessorR const &output,
                  GenericTensorAccessorW const &input) {
    size_t volume = get_num_elements(output.shape.dims).int_from_positive_int();
    cpu_cast_backward(
        output.get<IDT>(), input.get<ODT>(), volume, cast_to<ODT>(1.0f));
  }
};

void cpu_forward_kernel(GenericTensorAccessorR const &input,
                        GenericTensorAccessorW const &output) {
  DataTypeDispatch2<CPUForwardKernel>{}(
      input.shape.data_type, output.shape.data_type, input, output);
}

void cpu_backward_kernel(GenericTensorAccessorR const &output,
                         GenericTensorAccessorW const &input) {
  DataTypeDispatch2<CPUBackwardKernel>{}(
      output.shape.data_type, input.shape.data_type, output, input);
}

} // namespace FlexFlow::Kernels::Cast
