#include "caffe/layers/pooling_layer.hpp"
#include <chrono>

namespace caffe {


template <typename Dtype>
void PoolingLayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Forward_cpu(bottom, top);
  return;

  float2half(bottom[0]->count(), bottom[0]->cpu_data(), this->bottom_fp16_.mutable_cpu_data());
#ifdef PROFILE
    std::chrono::steady_clock::time_point begin_time = std::chrono::steady_clock::now();
#endif
    
  std::vector<void*> args = {(void*)this->bottom_fp16_.aicore_data(), (void*)this->top_fp16_.mutable_aicore_data()};
  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[0].kernel_, this->aicore_kernel_info_[0].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

#ifdef PROFILE
    std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
    std::cout << "Pooling time is " << std::chrono::duration_cast<std::chrono::milliseconds>(end_time - begin_time).count() << "[ms]" << std::endl;
#endif
  half2float(top[0]->count(), this->top_fp16_.cpu_data(), top[0]->mutable_cpu_data());

}
 
template <typename Dtype>
void PoolingLayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  Backward_cpu(top, propagate_down, bottom);
  return;


  float2half(top[0]->count(), top[0]->cpu_diff(), this->top_fp16_.mutable_cpu_diff());
  std::vector<void*> args;
  
    if(this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
        args.push_back((void*)this->bottom_fp16_.aicore_data());
        args.push_back((void*)this->top_fp16_.aicore_data());
    }
    args.push_back((void*)this->top_fp16_.aicore_diff());
    args.push_back((void*)this->bottom_fp16_.mutable_aicore_diff());

    AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[1].kernel_, this->aicore_kernel_info_[1].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
    AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));
    half2float(bottom[0]->count(), this->bottom_fp16_.cpu_diff(), bottom[0]->mutable_cpu_diff());            
}

INSTANTIATE_LAYER_AICORE_FUNCS(PoolingLayer);

}  // namespace caffe
