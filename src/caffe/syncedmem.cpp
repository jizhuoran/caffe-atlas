#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), gpu_ptr_(NULL), aicore_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), cpu_malloc_use_aicore_(false), own_gpu_data_(false), own_aicore_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), gpu_ptr_(NULL), aicore_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), cpu_malloc_use_aicore_(false), own_gpu_data_(false), own_aicore_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif 
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_, cpu_malloc_use_aicore_);
  }
#ifdef USE_AICORE
  if (aicore_ptr_ && own_aicore_data_) {
    AICORE_CHECK(rtFree(aicore_ptr_));
  }
#endif
#ifndef CPU_ONLY
  if (gpu_ptr_ && own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
#endif  // CPU_ONLY
}

inline void SyncedMemory::to_cpu() {
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, &cpu_malloc_use_aicore_);
    caffe_memset(size_, 0, cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, &cpu_malloc_use_aicore_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
    case HEAD_AT_AICORE:
#ifdef USE_AICORE
      if (cpu_ptr_ == NULL) {
        CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_, &cpu_malloc_use_aicore_);
        own_cpu_data_ = true;
      }
      AICORE_CHECK(rtMemcpy(cpu_ptr_, size_, aicore_ptr_, size_, RT_MEMCPY_DEVICE_TO_HOST));
      head_ = SYNCED;
#else
      NO_AICORE;
#endif
      break;
  case HEAD_AT_CPU:
  case SYNCED:
    break;
  }
}

inline void SyncedMemory::to_gpu() {
  check_device();
#ifndef CPU_ONLY
  switch (head_) {
  case UNINITIALIZED:
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    caffe_gpu_memset(size_, 0, gpu_ptr_);
    head_ = HEAD_AT_GPU;
    own_gpu_data_ = true;
    break;
  case HEAD_AT_CPU:
    if (gpu_ptr_ == NULL) {
      CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
      own_gpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, cpu_ptr_, gpu_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_GPU:
  case SYNCED:
    break;
  }
#else
  NO_GPU;
#endif
}

inline void SyncedMemory::to_aicore() {
#ifdef USE_AICORE
  switch (head_) {
  case UNINITIALIZED:
    AICORE_CHECK(rtMalloc(&aicore_ptr_, size_, RT_MEMORY_DDR));
    caffe_aicore_memset(size_, 0, aicore_ptr_);
    head_ = HEAD_AT_AICORE;
    own_aicore_data_ = true;
    break;
  case HEAD_AT_GPU:
    NO_GPU;
    break;
  case HEAD_AT_CPU:
    if (aicore_ptr_ == NULL) {
      AICORE_CHECK(rtMalloc(&aicore_ptr_, size_, RT_MEMORY_DDR));
      own_aicore_data_ = true;
    }
    AICORE_CHECK(rtMemcpy(aicore_ptr_, size_, cpu_ptr_, size_, RT_MEMCPY_HOST_TO_DEVICE));
    head_ = SYNCED;
    break;
  case HEAD_AT_AICORE:
  case SYNCED:
    break;
  }
#else
    NO_AICORE;
#endif
}


const void* SyncedMemory::cpu_data() {
  check_device();
  to_cpu();
  return (const void*)cpu_ptr_;
}

void SyncedMemory::set_cpu_data(void* data) {
  check_device();
  CHECK(data);
  if (own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_, cpu_malloc_use_aicore_);
  }
  cpu_ptr_ = data;
  head_ = HEAD_AT_CPU;
  own_cpu_data_ = false;
}

const void* SyncedMemory::gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  return (const void*)gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void SyncedMemory::set_gpu_data(void* data) {
  check_device();
#ifndef CPU_ONLY
  CHECK(data);
  if (own_gpu_data_) {
    CUDA_CHECK(cudaFree(gpu_ptr_));
  }
  gpu_ptr_ = data;
  head_ = HEAD_AT_GPU;
  own_gpu_data_ = false;
#else
  NO_GPU;
#endif
}

const void* SyncedMemory::aicore_data() {
#ifdef USE_AICORE
  to_aicore();
  return aicore_ptr_;
#else
  NO_AICORE;
  return NULL;
#endif
}

void* SyncedMemory::mutable_cpu_data() {
  check_device();
  to_cpu();
  head_ = HEAD_AT_CPU;
  return cpu_ptr_;
}

void* SyncedMemory::mutable_gpu_data() {
  check_device();
#ifndef CPU_ONLY
  to_gpu();
  head_ = HEAD_AT_GPU;
  return gpu_ptr_;
#else
  NO_GPU;
  return NULL;
#endif
}

void* SyncedMemory::mutable_aicore_data() {
#ifdef USE_AICORE
  to_aicore();
  head_ = HEAD_AT_AICORE;
  return aicore_ptr_;
#else
  NO_AICORE;
  return NULL;
#endif
}

#ifndef CPU_ONLY
void SyncedMemory::async_gpu_push(const cudaStream_t& stream) {
  check_device();
  CHECK(head_ == HEAD_AT_CPU);
  if (gpu_ptr_ == NULL) {
    CUDA_CHECK(cudaMalloc(&gpu_ptr_, size_));
    own_gpu_data_ = true;
  }
  const cudaMemcpyKind put = cudaMemcpyHostToDevice;
  CUDA_CHECK(cudaMemcpyAsync(gpu_ptr_, cpu_ptr_, size_, put, stream));
  // Assume caller will synchronize on the stream before use
  head_ = SYNCED;
}
#endif

void SyncedMemory::check_device() {
#ifndef CPU_ONLY
#ifdef DEBUG
  int device;
  cudaGetDevice(&device);
  CHECK(device == device_);
  if (gpu_ptr_ && own_gpu_data_) {
    cudaPointerAttributes attributes;
    CUDA_CHECK(cudaPointerGetAttributes(&attributes, gpu_ptr_));
    CHECK(attributes.device == device_);
  }
#endif
#endif
}

}  // namespace caffe

