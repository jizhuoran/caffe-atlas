#include "caffe/layers/conv_layer.hpp"

namespace caffe {

void ochw2fracZ(const float* ochw, _Float16* fracZ, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<_Float16 (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<const float (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          fracZ_array[(c_i/16) * (kernel_w * kernel_h) + h_i * (kernel_w) + w_i][(o_i)/16][o_i%16][c_i%16] = ochw_array[o_i][c_i][h_i][w_i];
        }
      }
    }
  }
}

void ochw2fracZ(const double* ochw, _Float16* fracZ, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<_Float16 (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<const double (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          fracZ_array[(c_i/16) * (kernel_w * kernel_h) + h_i * (kernel_w) + w_i][(o_i)/16][o_i%16][c_i%16] = ochw_array[o_i][c_i][h_i][w_i];
        }
      }
    }
  }
}

void ochw2fracZ(const _Float16* ochw, _Float16* fracZ, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<_Float16 (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<const _Float16 (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          fracZ_array[(c_i/16) * (kernel_w * kernel_h) + h_i * (kernel_w) + w_i][(o_i)/16][o_i%16][c_i%16] = ochw_array[o_i][c_i][h_i][w_i];
        }
      }
    }
  }
}

void fracZ2ochw(const float* fracZ, float* ochw, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<const float (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<float (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          ochw_array[o_i][c_i][h_i][w_i] = fracZ_array[(c_i/16) * (kernel_w * kernel_h) + h_i * (kernel_w) + w_i][(o_i)/16][o_i%16][c_i%16];
        }
      }
    }
  }
}

void fracZ2ochw(const float* fracZ, double* ochw, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<const float (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<double (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          ochw_array[o_i][c_i][h_i][w_i] = fracZ_array[(c_i/16) * (kernel_w * kernel_h) + h_i * (kernel_w) + w_i][(o_i)/16][o_i%16][c_i%16];
        }
      }
    }
  }
}

void fracZ2ochw(const float* fracZ, _Float16* ochw, int channel_out, int channel_in, int kernel_h, int kernel_w) {
  auto fracZ_array = *reinterpret_cast<const float (*)[kernel_h * kernel_w * ((channel_in + 15) / 16)][(channel_out + 15) / 16][16][16]>(fracZ);
  auto ochw_array = *reinterpret_cast<_Float16 (*)[channel_out][channel_in][kernel_h][kernel_w]>(ochw);
  for (int o_i = 0; o_i < channel_out; o_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < kernel_h; h_i++) {
        for (int w_i = 0; w_i < kernel_w; w_i++) {
          ochw_array[o_i][c_i][h_i][w_i] = fracZ_array[(c_i/16) * (kernel_w * kernel_h) + h_i * (kernel_w) + w_i][(o_i)/16][o_i%16][c_i%16];
        }
      }
    }
  }
}



void five2four(const _Float16* five, float* four, int batch_size, int channel_in, int in_height, int in_width) {
  auto five_array = *reinterpret_cast<const _Float16 (*)[batch_size][(channel_in+15)/16][in_height][in_width][16]>(five);
  auto four_array = *reinterpret_cast<float (*)[batch_size][channel_in][in_height][in_width]>(four);
  
  for (int n_i = 0; n_i < batch_size; n_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < in_height; h_i++) {
        for (int w_i = 0; w_i < in_width; w_i++) {
          four_array[n_i][c_i][h_i][w_i] = five_array[n_i][c_i/16][h_i][w_i][c_i%16];
        }
      }
    }
  }
}

void five2four(const _Float16* five, double* four, int batch_size, int channel_in, int in_height, int in_width) {
  auto five_array = *reinterpret_cast<const _Float16 (*)[batch_size][(channel_in+15)/16][in_height][in_width][16]>(five);
  auto four_array = *reinterpret_cast<double (*)[batch_size][channel_in][in_height][in_width]>(four);
  
  for (int n_i = 0; n_i < batch_size; n_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < in_height; h_i++) {
        for (int w_i = 0; w_i < in_width; w_i++) {
          four_array[n_i][c_i][h_i][w_i] = five_array[n_i][c_i/16][h_i][w_i][c_i%16];
        }
      }
    }
  }
}

void four2five(const float* four, _Float16* five, int batch_size, int channel_in, int in_height, int in_width) {
  auto five_array = *reinterpret_cast<_Float16 (*)[batch_size][(channel_in+15)/16][in_height][in_width][16]>(five);
  auto four_array = *reinterpret_cast<const float (*)[batch_size][channel_in][in_height][in_width]>(four);
  for (int n_i = 0; n_i < batch_size; n_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < in_height; h_i++) {
        for (int w_i = 0; w_i < in_width; w_i++) {
          five_array[n_i][c_i/16][h_i][w_i][c_i%16] = four_array[n_i][c_i][h_i][w_i];
        }
      }
    }
  }
}

void four2five(const double* four, _Float16* five, int batch_size, int channel_in, int in_height, int in_width) {
  auto five_array = *reinterpret_cast<_Float16 (*)[batch_size][(channel_in+15)/16][in_height][in_width][16]>(five);
  auto four_array = *reinterpret_cast<const double (*)[batch_size][channel_in][in_height][in_width]>(four);
  for (int n_i = 0; n_i < batch_size; n_i++) {
    for (int c_i = 0; c_i < channel_in; c_i++) {
      for (int h_i = 0; h_i < in_height; h_i++) {
        for (int w_i = 0; w_i < in_width; w_i++) {
          five_array[n_i][c_i/16][h_i][w_i][c_i%16] = four_array[n_i][c_i][h_i][w_i];
        }
      }
    }
  }
}



template <typename Dtype>
void ConvolutionLayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  four2five(bottom[0]->cpu_data(), this->bottom_five_fp16_.mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));

  ochw2fracZ(this->blobs_[0]->cpu_data(), this->fracZ_fp16_.mutable_cpu_data(), this->num_output_, this->channels_, this->blobs_[0]->shape(2), this->blobs_[0]->shape(3));

  std::vector<void*> args = {(void*)this->bottom_five_fp16_.aicore_data(), (void*)this->fracZ_fp16_.aicore_data()};
  std::unique_ptr<Blob<_Float16>> bias_fp16;
  
  if (this->bias_term_) {
    auto bias_fp16_data = this->bias_fp16_.mutable_cpu_data();
    auto bias_data = this->blobs_[1]->cpu_data();
    for(int i = 0; i < this->blobs_[1]->count(); ++i) {
      bias_fp16_data[i] = bias_data[i];
    }
    args.push_back((void*)this->bias_fp16_.aicore_data());
  }
  args.push_back((void*)this->top_five_fp16_.mutable_aicore_data());

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[0].kernel_, this->aicore_kernel_info_[0].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

  five2four(this->top_five_fp16_.cpu_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

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

  four2five(top[0]->cpu_diff(), this->top_five_fp16_.mutable_cpu_diff(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

  std::vector<void*> args = { (void*)this->bottom_five_fp16_.aicore_data(), 
                              (void*)this->top_five_fp16_.aicore_diff(),
                              (void*)this->fracZ_fp32_.mutable_aicore_diff()};


  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[1].kernel_, this->aicore_kernel_info_[1].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

  fracZ2ochw(this->fracZ_fp32_.cpu_diff(), this->blobs_[0]->mutable_cpu_diff(), this->num_output_, this->channels_, this->blobs_[0]->shape(2), this->blobs_[0]->shape(3));
  

  std::vector<void*> args1 = { (void*)this->fracZ_fp16_.aicore_data(), 
                              (void*)this->top_five_fp16_.aicore_diff(),
                              (void*)this->bottom_five_fp16_.mutable_aicore_diff()};

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[2].kernel_, this->aicore_kernel_info_[2].block_num_, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));
  five2four(this->bottom_five_fp16_.cpu_diff(), bottom[0]->mutable_cpu_diff(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));


}

INSTANTIATE_LAYER_AICORE_FUNCS(ConvolutionLayer);

}  // namespace caffe
