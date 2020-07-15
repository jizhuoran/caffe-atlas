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
  

  if (this->kernel_shape_.cpu_data()[0] == 1 && this->kernel_shape_.cpu_data()[1] == 1) {
    Forward_cpu(bottom, top);
    return;
  }

  if (this->stride_.cpu_data()[0] != 1 || this->stride_.cpu_data()[0] != 1) {
    Forward_cpu(bottom, top);
    return;
  }


  //auto weight_fraz = ochw2fracZ<Dtype>(this->blobs_[0].get(), this->num_output_, this->channels_, this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1]);
  //Blob<Dtype>* weight = this->blobs_[0].get();
  Blob<Dtype> bottom_five(bottom[0]->shape(0), (bottom[0]->shape(1)+15)/16*16, bottom[0]->shape(2), bottom[0]->shape(3));
  four2five(bottom[0]->cpu_data(), bottom_five.mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));


  std::vector<int> top_five_shape{top[0]->shape(0), (top[0]->shape(1) + 15) / 16, top[0]->shape(2), top[0]->shape(3) * 16};
  Blob<Dtype> top_five(top_five_shape);
  
  auto hack_str = new std::string(fmt::format("{}__kernel0", kernel_identifier()));

  //std::vector<std::string> input_datas = {bottom_five.aicore_data(), weight_fraz->aicore_data()};
  std::vector<std::string> input_datas = {bottom_five.aicore_data(), this->blobs_[0]->aicore_data()};
  //std::cout<<this->blobs_[0]->aicore_data()<<std::endl;
  if (this->bias_term_) {
    input_datas.push_back(this->blobs_[1]->aicore_data());
  }
  auto err = custom::op_run(*hack_str, 
                                   0,
                                   fmt::format("{}/{}.o", Caffe::kernel_dir(), kernel_identifier()),
                                   input_datas,
                                   {top_five.aicore_data()},
                                   {top_five.count() * static_cast<unsigned int>(sizeof(half))});
  AICORE_CHECK(err);
  five2four(top_five.cpu_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {


  if (this->kernel_shape_.cpu_data()[0] == 1 && this->kernel_shape_.cpu_data()[1] == 1) {
    Backward_cpu(top, propagate_down, bottom);
    return;
  }

  if (this->stride_.cpu_data()[0] != 1 || this->stride_.cpu_data()[1] != 1) {
    Backward_cpu(top, propagate_down, bottom);
    return;
  }


  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  
  
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + n * this->top_dim_);
      }
    }










  Blob<Dtype> bottom_five(bottom[0]->shape(0), (bottom[0]->shape(1)+15)/16*16, bottom[0]->shape(2), bottom[0]->shape(3));
  four2five(bottom[0]->cpu_data(), bottom_five.mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));

  Blob<Dtype> top_five(top[0]->shape(0), (top[0]->shape(1)+15)/16*16, top[0]->shape(2), top[0]->shape(3));
  four2five(top[0]->cpu_diff(), top_five.mutable_cpu_diff(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

  //auto weight_fraz = ochw2fracZ<Dtype>(this->blobs_[0].get(), this->num_output_, this->channels_, this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1]);



  auto hack_str_weight = new std::string(fmt::format("conv_bw_weight_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}__kernel0", 
                                                this->num_,
                                                this->channels_,
                                                this->num_output_,
                                                this->input_shape(1),
                                                this->input_shape(2),
                                                this->kernel_shape_.cpu_data()[0],
                                                this->kernel_shape_.cpu_data()[1],
                                                this->pad_.cpu_data()[0],
                                                this->pad_.cpu_data()[1],
                                                this->stride_.cpu_data()[0],
                                                this->stride_.cpu_data()[1]));

  auto weight_fraz = this->blobs_[0].get();
  auto weight_fraz_diff_32 = Caffe::aicore_dir() + "/" + (*hack_str_weight);

  auto err_weight = custom::op_run(*hack_str_weight, 
                                   0,
                                   fmt::format("{}/conv_bw_weight_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.o", Caffe::kernel_dir(),
                                                this->num_,
                                                this->channels_,
                                                this->num_output_,
                                                this->input_shape(1),
                                                this->input_shape(2),
                                                this->kernel_shape_.cpu_data()[0],
                                                this->kernel_shape_.cpu_data()[1],
                                                this->pad_.cpu_data()[0],
                                                this->pad_.cpu_data()[1],
                                                this->stride_.cpu_data()[0],
                                                this->stride_.cpu_data()[1]),
                                   {bottom_five.aicore_data(), top_five.aicore_diff()},
                                   {weight_fraz_diff_32},
                                   {weight_fraz->count() * static_cast<unsigned int>(sizeof(float))}); //!!! THIS MUST BE FP32
  caffe_aicore_memcpy(weight_fraz->count() * static_cast<unsigned int>(sizeof(float)), weight_fraz_diff_32, weight_fraz->mutable_cpu_diff());
  std::remove(weight_fraz_diff_32.c_str());

  //fracZ2ochw(weight_fraz->cpu_diff(), this->blobs_[0]->mutable_cpu_diff(), this->num_output_, this->channels_, this->kernel_shape_.cpu_data()[0], this->kernel_shape_.cpu_data()[1]);

  AICORE_CHECK(err_weight);


  // DO WEIGHT



  auto hack_str = new std::string(fmt::format("conv_bw_input_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}__kernel0", 
                                                this->num_,
                                                this->channels_,
                                                this->num_output_,
                                                this->input_shape(1),
                                                this->input_shape(2),
                                                this->kernel_shape_.cpu_data()[0],
                                                this->kernel_shape_.cpu_data()[1],
                                                this->pad_.cpu_data()[0],
                                                this->pad_.cpu_data()[1],
                                                this->stride_.cpu_data()[0],
                                                this->stride_.cpu_data()[1]));

  auto err = custom::op_run(*hack_str, 
                                   0,
                                   fmt::format("{}/conv_bw_input_op_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.o", Caffe::kernel_dir(),
                                                this->num_,
                                                this->channels_,
                                                this->num_output_,
                                                this->input_shape(1),
                                                this->input_shape(2),
                                                this->kernel_shape_.cpu_data()[0],
                                                this->kernel_shape_.cpu_data()[1],
                                                this->pad_.cpu_data()[0],
                                                this->pad_.cpu_data()[1],
                                                this->stride_.cpu_data()[0],
                                                this->stride_.cpu_data()[1]),
                                   {weight_fraz->aicore_data(), top_five.aicore_diff()},
                                   {bottom_five.mutable_aicore_diff()},
                                   {bottom_five.count() * static_cast<unsigned int>(sizeof(half))});

  AICORE_CHECK(err);
  five2four(bottom_five.cpu_diff(), bottom[0]->mutable_cpu_diff(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));

}

INSTANTIATE_LAYER_AICORE_FUNCS(ConvolutionLayer);

}  // namespace caffe
