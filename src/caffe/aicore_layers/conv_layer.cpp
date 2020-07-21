#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/conv_layer.hpp"
#include "caffe/util/half.hpp"
#include "caffe/util/math_functions.hpp"

#include <numeric>


namespace caffe {


template <typename Dtype>
std::shared_ptr<Blob<Dtype>> ochw2fracZ(Blob<Dtype>* ochw, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  std::vector<int> fracZ_shape{ kernel_h * kernel_w * ((channel_in + 15) / 16) , (channel_out + 15) / 16 ,16, 16};
  std::shared_ptr<Blob<Dtype>> fracZ = std::make_shared<Blob<Dtype>>(fracZ_shape[0], fracZ_shape[1], fracZ_shape[2], fracZ_shape[3]);
  auto fracZ_data = *reinterpret_cast<Dtype (*)[fracZ_shape[0]][fracZ_shape[1]][fracZ_shape[2]][fracZ_shape[3]]>(fracZ->mutable_cpu_data());
  auto ochw_data = *reinterpret_cast<const Dtype (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw->cpu_data());
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          fracZ_data[(h_i * kernel_w + w_i) * ((channel_in+15)/16) + (c_i/16)][(o_i)/16][o_i%16][c_i%16] = ochw_data[o_i][c_i][h_i][w_i];
        }
      }
    }
  }
  return fracZ;
}

template <typename Dtype>
void fracZ2ochw(const Dtype* fracZ, Dtype* ochw, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<const Dtype (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<Dtype (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          ochw_array[o_i][c_i][h_i][w_i] = fracZ_array[(h_i * kernel_w + w_i) * ((channel_in+15)/16) + (c_i/16)][(o_i)/16][o_i%16][c_i%16];
        }
      }
    }
  }
}

template <typename Dtype>
std::string ConvolutionLayer<Dtype>::kernel_identifier() {
  return fmt::format("conv_fw_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}", 
                      this->num_,
                      this->channels_,
                      this->num_output_,
                      this->input_shape(1),
                      this->input_shape(2),
                      (this->bias_term_? "bias" : "nobias"),
                      this->kernel_shape_.cpu_data()[0],
                      this->kernel_shape_.cpu_data()[1],
                      this->pad_.cpu_data()[0],
                      this->pad_.cpu_data()[1],
                      this->stride_.cpu_data()[0],
                      this->stride_.cpu_data()[1]);
}


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
  args.push_back((void*)top_five.aicore_data());

  AICORE_CHECK(rtKernelLaunch(this->fw_kernel, this->fw_block_num, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

  five2four(top_five.aicore_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

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

  // AICORE_CHECK(rtKernelLaunch(this->bw_weight_kernel, this->bw_weight_block_num, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  // AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

  const float* cpu_weight_fraz_diff_fp32 = weight_fraz_diff_fp32.aicore_diff();
  for(int i = 0; i < weight_fraz->count(); ++i) {
    weight_fraz->mutable_cpu_diff()[i] = Dtype(cpu_weight_fraz_diff_fp32[i]);
  }

  // DO WEIGHT

  std::vector<void*> args1 = { (void*)weight_fraz->aicore_data(), 
                              (void*)top_five.aicore_diff(),
                              (void*)bottom_five.mutable_aicore_diff()};

  AICORE_CHECK(rtKernelLaunch(this->bw_input_kernel, this->bw_input_block_num, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));
  five2four(bottom_five.aicore_diff(), bottom[0]->mutable_cpu_diff(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));


}

INSTANTIATE_LAYER_AICORE_FUNCS(ConvolutionLayer);

}  // namespace caffe
