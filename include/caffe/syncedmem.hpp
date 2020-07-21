#ifndef CAFFE_SYNCEDMEM_HPP_
#define CAFFE_SYNCEDMEM_HPP_

#include <cstdlib>

#ifdef USE_MKL
  #include "mkl.h"
#endif

#include "caffe/common.hpp"

namespace caffe {

// If CUDA is available and in GPU mode, host memory will be allocated pinned,
// using cudaMallocHost. It avoids dynamic pinning for transfers (DMA).
// The improvement in performance seems negligible in the single GPU case,
// but might be more significant for parallel training. Most importantly,
// it improved stability for large models on many GPUs.
inline void CaffeMallocHost(void** ptr, size_t size, bool* use_cuda, bool* use_aicore) {
#ifndef CPU_ONLY
  if (Caffe::mode() == Caffe::GPU) {
    CUDA_CHECK(cudaMallocHost(ptr, size));
    *use_cuda = true;
    return;
  }
#endif
  if (Caffe::mode() == Caffe::AICORE) {
    AICORE_CHECK(rtMallocHost(ptr, size? size:4));
    *use_aicore = true;
    return;
  } else {
#ifdef USE_MKL
    *ptr = mkl_malloc(size ? size:1, 64);
#else
    *ptr = malloc(size);
#endif
    *use_cuda = false;
  }
  CHECK(*ptr) << "host allocation of size " << size << " failed";
}

inline void CaffeFreeHost(void* ptr, bool use_cuda, bool use_aicore) {
#ifndef CPU_ONLY
  if (use_cuda) {
    CUDA_CHECK(cudaFreeHost(ptr));
    return;
  }
#endif
  if (use_aicore) {
    AICORE_CHECK(rtFreeHost(ptr));
  } else {
#ifdef USE_MKL
    mkl_free(ptr);
#else
    free(ptr);
#endif
  }
}


/**
 * @brief Manages memory allocation and synchronization between the host (CPU)
 *        and device (GPU).
 *
 * TODO(dox): more thorough description.
 */
class SyncedMemory {
 public:
  SyncedMemory();
  explicit SyncedMemory(size_t size);
  ~SyncedMemory();
  const void* cpu_data();
  void set_cpu_data(void* data);
  const void* gpu_data();
  void set_gpu_data(void* data);
  const void* aicore_data();
  void* mutable_cpu_data();
  void* mutable_gpu_data();
  void* mutable_aicore_data();
  enum SyncedHead { UNINITIALIZED, HEAD_AT_CPU, HEAD_AT_GPU, HEAD_AT_AICORE, SYNCED };
  SyncedHead head() const { return head_; }
  size_t size() const { return size_; }

#ifndef CPU_ONLY
  void async_gpu_push(const cudaStream_t& stream);
#endif

 private:
  void check_device();

  void to_cpu();
  void to_gpu();
  void to_aicore();
  void* cpu_ptr_;
  void* gpu_ptr_;
  void* aicore_ptr_;
  size_t size_;
  SyncedHead head_;
  bool own_cpu_data_;
  bool cpu_malloc_use_cuda_;
  bool own_gpu_data_;
  bool cpu_malloc_use_aicore_;
  bool own_aicore_data_;
  int device_;

  DISABLE_COPY_AND_ASSIGN(SyncedMemory);
};  // class SyncedMemory

}  // namespace caffe

#endif  // CAFFE_SYNCEDMEM_HPP_
