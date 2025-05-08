#include "kernels/managed_per_device_ff_handle.h"
#include "internal/device.h"
#include "kernels/nccl.h"

namespace FlexFlow {

ManagedPerDeviceFFHandle::ManagedPerDeviceFFHandle(
    int num_ranks, int my_rank, size_t workSpaceSize, bool allowTensorOpMathConversion) {
  this->handle = new PerDeviceFFHandle{};
  this->handle->workSpaceSize = workSpaceSize;
  this->handle->allowTensorOpMathConversion = allowTensorOpMathConversion;

  checkCUDNN(cudnnCreate(&this->handle->dnn));
  checkCUBLAS(cublasCreate(&this->handle->blas));
  checkCUDA(cudaMalloc(&this->handle->workSpace, this->handle->workSpaceSize));

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
  if (this->handle != nullptr) {
    checkCUDNN(cudnnDestroy(this->handle->dnn));
    checkCUBLAS(cublasDestroy(this->handle->blas));
    checkCUDA(cudaFree(this->handle->workSpace));
#ifdef FF_USE_NCCL
    checkNCCL(ncclCommDestroy(this->handle->ncclComm));
#endif
    delete this->handle;
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
