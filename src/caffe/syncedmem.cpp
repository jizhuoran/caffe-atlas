#include "caffe/common.hpp"
#include "caffe/syncedmem.hpp"
#include "caffe/util/half.hpp"
#include "caffe/util/math_functions.hpp"
#include <string>
#include <unistd.h>
#include <vector>

namespace caffe {
SyncedMemory::SyncedMemory()
  : cpu_ptr_(NULL), debug_cpu_ptr_(NULL), gpu_ptr_(NULL), size_(0), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::SyncedMemory(size_t size)
  : cpu_ptr_(NULL), debug_cpu_ptr_(NULL), gpu_ptr_(NULL), size_(size), head_(UNINITIALIZED),
    own_cpu_data_(false), cpu_malloc_use_cuda_(false), own_gpu_data_(false) {
#ifndef CPU_ONLY
#ifdef DEBUG
  CUDA_CHECK(cudaGetDevice(&device_));
#endif
#endif
}

SyncedMemory::~SyncedMemory() {
  check_device();
  if (cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
  }
  if (debug_cpu_ptr_ && own_cpu_data_) {
    CaffeFreeHost(debug_cpu_ptr_, cpu_malloc_use_cuda_);
  }
  if (aicore_ptr_ != "") {
    std::remove(aicore_ptr_.c_str());
  }

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
    CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
    CaffeMallocHost(&debug_cpu_ptr_, size_/2, &cpu_malloc_use_cuda_);
    caffe_memset(size_, 0, cpu_ptr_);
    caffe_memset(size_/2, 0, debug_cpu_ptr_);
    head_ = HEAD_AT_CPU;
    own_cpu_data_ = true;
    break;
  case HEAD_AT_GPU:
#ifndef CPU_ONLY
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_gpu_memcpy(size_, gpu_ptr_, cpu_ptr_);
    head_ = SYNCED;
#else
    NO_GPU;
#endif
    break;
  case HEAD_AT_AICORE:
    if (cpu_ptr_ == NULL) {
      CaffeMallocHost(&cpu_ptr_, size_, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    if (debug_cpu_ptr_ == NULL) {
      CaffeMallocHost(&debug_cpu_ptr_, size_/2, &cpu_malloc_use_cuda_);
      own_cpu_data_ = true;
    }
    caffe_aicore_memcpy(size_/2, aicore_ptr_, debug_cpu_ptr_);
    half2float(size_ / 4, reinterpret_cast<half*>(debug_cpu_ptr_), reinterpret_cast<float*>(cpu_ptr_));
    head_ = SYNCED;
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
  check_device();
  switch (head_) {
  case UNINITIALIZED:
    assert(aicore_ptr_ == "" && "The aicore_ptr_ should be null if UNINITIALIZED");
    aicore_ptr_ = Caffe::aicore_dir() + std::to_string(reinterpret_cast<unsigned long long int>(this)); //FIX ME: only valid for 64bit machine
    caffe_aicore_set(size_/4, .0, aicore_ptr_); //FIX_ME
    head_ = HEAD_AT_AICORE;
    break;
  case HEAD_AT_GPU:
    NO_GPU;
    break;
  case HEAD_AT_CPU:
    if (aicore_ptr_ == "") {
      aicore_ptr_ = Caffe::aicore_dir() + std::to_string(reinterpret_cast<unsigned long long int>(this)); //FIX ME: only valid for 64bit machine
    }
    float2half(size_ / 4, reinterpret_cast<float*>(cpu_ptr_), reinterpret_cast<half*>(debug_cpu_ptr_));
    caffe_aicore_memcpy(size_/2, debug_cpu_ptr_, aicore_ptr_);
    head_ = SYNCED;
    break;
  case HEAD_AT_AICORE:
  case SYNCED:
    break;
  }
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
    CaffeFreeHost(cpu_ptr_, cpu_malloc_use_cuda_);
    CaffeFreeHost(debug_cpu_ptr_, cpu_malloc_use_cuda_);
  }
  cpu_ptr_ = data;
  CaffeMallocHost(&debug_cpu_ptr_, size_/2, &cpu_malloc_use_cuda_); //UGLY
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

std::string SyncedMemory::aicore_data() {
  check_device();
  to_aicore();
  return aicore_ptr_;
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

std::string SyncedMemory::mutable_aicore_data() {
  check_device();
  to_aicore();
  head_ = HEAD_AT_AICORE;
  return aicore_ptr_;
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

