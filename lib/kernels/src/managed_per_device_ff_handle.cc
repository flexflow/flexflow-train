#include "kernels/managed_per_device_ff_handle.h"
#include "device.h"
#include "kernels/nccl.h"

namespace FlexFlow {

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle(int num_ranks, int my_rank) {
  handle = new PerDeviceFFHandle;
  handle->workSpaceSize = 1024 * 1024;
  handle->allowTensorOpMathConversion = true;

  checkCUDNN(cudnnCreate(&handle->dnn));
  checkCUBLAS(cublasCreate(&handle->blas));
  checkCUDA(cudaMalloc(&handle->workSpace, handle->workSpaceSize));

#ifdef FF_USE_NCCL
  ncclUniqueId ncclId;
  checkNCCL(ncclGetUniqueId(&ncclId));
  checkNCCL(ncclCommInitRank(
      &handle->ncclComm, num_ranks, ncclId, my_rank)); // todo generalize
#endif
}

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle(
    ManagedPerDeviceFFHandle &&other) noexcept
    : handle(std::exchange(other.handle, nullptr)) {}

ManagedPerDeviceFFHandle &ManagedPerDeviceFFHandle::operator=(
    ManagedPerDeviceFFHandle &&other) noexcept {
  std::swap(this->handle, other.handle);
  return *this;
}

ManagedPerDeviceFFHandle::~ManagedPerDeviceFFHandle() {
  if (handle != nullptr) {
    checkCUDNN(cudnnDestroy(handle->dnn));
    checkCUBLAS(cublasDestroy(handle->blas));
    checkCUDA(cudaFree(handle->workSpace));
#ifdef FF_USE_NCCL
    checkNCCL(ncclCommDestroy(handle->ncclComm));
#endif
    delete handle;
  }
}

PerDeviceFFHandle const &ManagedPerDeviceFFHandle::raw_handle() const {
  return *handle;
}

ManagedPerDeviceFFHandle initialize_single_gpu_handle() {
  return ManagedPerDeviceFFHandle(1, 0);
}

ManagedPerDeviceFFHandle initialize_multi_gpu_handle(int num_ranks,
                                                     int my_rank) {
  return ManagedPerDeviceFFHandle(num_ranks, my_rank);
}

} // namespace FlexFlow
