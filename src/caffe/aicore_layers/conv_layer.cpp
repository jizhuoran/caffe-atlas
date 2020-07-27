#include "caffe/layers/conv_layer.hpp"

namespace caffe {

template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  Blob<Dtype> bottom_five(bottom[0]->shape(0), (bottom[0]->shape(1)+15)/16*16, bottom[0]->shape(2), bottom[0]->shape(3));
  four2five(bottom[0]->cpu_data(), bottom_five.mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));

  std::vector<int> top_five_shape{top[0]->shape(0), (top[0]->shape(1) + 15) / 16, top[0]->shape(2), top[0]->shape(3) * 16};
  Blob<Dtype> top_five(top_five_shape);

  std::vector<void*> args = {(void*)bottom_five.aicore_data(), (void*)this->blobs_[0]->aicore_data()};
  if (this->bias_term_) {
    args.push_back((void*)this->blobs_[1]->aicore_data());
  }
  args.push_back((void*)top_five.mutable_aicore_data());

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[0].kernel_, this->aicore_kernel_info_[0].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

  five2four(top_five.cpu_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  // Bias gradient, if necessary.
  if (this->bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
    for (int i = 0; i < this->num_; ++i) {
      for(int m = 0; m < this->num_output_; ++m) {
        for(int n = 0; n < this->out_spatial_dim_; ++n) {
          bias_diff[m] += top_diff[m * this->out_spatial_dim_ + n];
        }
      }
      top_diff += this->top_dim_;
    }
  }


  Blob<Dtype> bottom_five(bottom[0]->shape(0), (bottom[0]->shape(1)+15)/16*16, bottom[0]->shape(2), bottom[0]->shape(3));
  four2five(bottom[0]->cpu_data(), bottom_five.mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));

  Blob<Dtype> top_five(top[0]->shape(0), (top[0]->shape(1)+15)/16*16, top[0]->shape(2), top[0]->shape(3));
  four2five(top[0]->cpu_diff(), top_five.mutable_cpu_diff(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));


  auto weight_fraz = this->blobs_[0].get();

  Blob<float> weight_fraz_diff_fp32({weight_fraz->count()});
  std::vector<void*> args = { (void*)bottom_five.aicore_data(), 
                              (void*)top_five.aicore_diff(),
                              (void*)weight_fraz_diff_fp32.mutable_aicore_diff()};

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[1].kernel_, this->aicore_kernel_info_[1].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

  const float* cpu_weight_fraz_diff_fp32 = weight_fraz_diff_fp32.cpu_diff();
  for(int i = 0; i < weight_fraz->count(); ++i) {
    weight_fraz->mutable_cpu_diff()[i] = Dtype(cpu_weight_fraz_diff_fp32[i]);
  }

  std::vector<void*> args1 = { (void*)weight_fraz->aicore_data(), 
                              (void*)top_five.aicore_diff(),
                              (void*)bottom_five.mutable_aicore_diff()};

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[2].kernel_, this->aicore_kernel_info_[2].block_num_, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));
  five2four(bottom_five.aicore_diff(), bottom[0]->mutable_cpu_diff(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));


}

INSTANTIATE_LAYER_AICORE_FUNCS(ConvolutionLayer);

}  // namespace caffe
