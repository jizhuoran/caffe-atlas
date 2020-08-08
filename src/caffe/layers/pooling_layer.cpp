#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/layers/pooling_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

using std::min;
using std::max;

template <typename Dtype>
void PoolingLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

#ifdef USE_AICORE
  // if(Caffe::aicore_mode()) {
  //   AicoreKerel fw_param = this->layer_param_.aicorekernel(0);
  //   char* fw_stub = Caffe::Get().new_load_aicore_kernel(fw_param.kernelfile(), fw_param.kernelname());
  //   this->aicore_kernel_info_.push_back(AICoreKernelInfo(fw_stub, fw_param.block_num()));

  //   AicoreKerel bw_input_param = this->layer_param_.aicorekernel(1);
  //   char* bw_input_stub = Caffe::Get().new_load_aicore_kernel(bw_input_param.kernelfile(), bw_input_param.kernelname());
  //   this->aicore_kernel_info_.push_back(AICoreKernelInfo(bw_input_stub, bw_input_param.block_num()));
  // }
#endif

  PoolingParameter pool_param = this->layer_param_.pooling_param();
  if (pool_param.global_pooling()) {
    CHECK(!(pool_param.has_kernel_size() ||
      pool_param.has_kernel_h() || pool_param.has_kernel_w()))
      << "With Global_pooling: true Filter size cannot specified";
  } else {
    CHECK(!pool_param.has_kernel_size() !=
      !(pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "Filter size is kernel_size OR kernel_h and kernel_w; not both";
    CHECK(pool_param.has_kernel_size() ||
      (pool_param.has_kernel_h() && pool_param.has_kernel_w()))
      << "For non-square filters both kernel_h and kernel_w are required.";
  }
  CHECK((!pool_param.has_pad() && pool_param.has_pad_h()
      && pool_param.has_pad_w())
      || (!pool_param.has_pad_h() && !pool_param.has_pad_w()))
      << "pad is pad OR pad_h and pad_w are required.";
  CHECK((!pool_param.has_stride() && pool_param.has_stride_h()
      && pool_param.has_stride_w())
      || (!pool_param.has_stride_h() && !pool_param.has_stride_w()))
      << "Stride is stride OR stride_h and stride_w are required.";
  global_pooling_ = pool_param.global_pooling();
  round_mode_ = pool_param.round_mode();
  if (global_pooling_) {
    kernel_h_ = bottom[0]->shape(2);
    kernel_w_ = bottom[0]->shape(3);
  } else {
    if (pool_param.has_kernel_size()) {
      kernel_h_ = kernel_w_ = pool_param.kernel_size();
    } else {
      kernel_h_ = pool_param.kernel_h();
      kernel_w_ = pool_param.kernel_w();
    }
  }
  CHECK_GT(kernel_h_, 0) << "Filter dimensions cannot be zero.";
  CHECK_GT(kernel_w_, 0) << "Filter dimensions cannot be zero.";
  if (!pool_param.has_pad_h()) {
    pad_h_ = pad_w_ = pool_param.pad();
  } else {
    pad_h_ = pool_param.pad_h();
    pad_w_ = pool_param.pad_w();
  }
  if (!pool_param.has_stride_h()) {
    stride_h_ = stride_w_ = pool_param.stride();
  } else {
    stride_h_ = pool_param.stride_h();
    stride_w_ = pool_param.stride_w();
  }
  if (global_pooling_) {
    CHECK(pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1)
      << "With Global_pooling: true; only pad = 0 and stride = 1";
  }
  if (pad_h_ != 0 || pad_w_ != 0) {
    CHECK(this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_AVE
        || this->layer_param_.pooling_param().pool()
        == PoolingParameter_PoolMethod_MAX)
        << "Padding implemented only for average and max pooling.";
    CHECK_LT(pad_h_, kernel_h_);
    CHECK_LT(pad_w_, kernel_w_);
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(5, bottom[0]->num_axes()) << "Input must have 4 axes, "
      << "corresponding to (num, channels, height, width)";
  channels_ = bottom[0]->shape(1);
  height_ = bottom[0]->shape(2);
  width_ = bottom[0]->shape(3);
  if (global_pooling_) {
    kernel_h_ = bottom[0]->shape(2);
    kernel_w_ = bottom[0]->shape(3);
  }
  pool_size_ = kernel_h_ * kernel_w_;
  switch (round_mode_) {
  case PoolingParameter_RoundMode_CEIL:
    pooled_height_ = static_cast<int>(ceil(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(ceil(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    break;
  case PoolingParameter_RoundMode_FLOOR:
    pooled_height_ = static_cast<int>(floor(static_cast<float>(
        height_ + 2 * pad_h_ - kernel_h_) / stride_h_)) + 1;
    pooled_width_ = static_cast<int>(floor(static_cast<float>(
        width_ + 2 * pad_w_ - kernel_w_) / stride_w_)) + 1;
    break;
  default:
    LOG(FATAL) << "Unknown rounding mode.";
  }
  if (pad_h_ || pad_w_) {
    // If we have padding, ensure that the last pooling starts strictly
    // inside the image (instead of at the padding); otherwise clip the last.
    if ((pooled_height_ - 1) * stride_h_ >= height_ + pad_h_) {
      --pooled_height_;
    }
    if ((pooled_width_ - 1) * stride_w_ >= width_ + pad_w_) {
      --pooled_width_;
    }
    CHECK_LT((pooled_height_ - 1) * stride_h_, height_ + pad_h_);
    CHECK_LT((pooled_width_ - 1) * stride_w_, width_ + pad_w_);
  }
  top[0]->Reshape(std::vector<int>{bottom[0]->shape(0), channels_, pooled_height_,
      pooled_width_, 16});
  if (top.size() > 1) {
    top[1]->ReshapeLike(*top[0]);
  }
  // If max pooling, we will initialize the vector index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_MAX && top.size() == 1) {
    max_idx_.Reshape(std::vector<int>{bottom[0]->shape(0), channels_, pooled_height_,
        pooled_width_, 16});
  }
  // If stochastic pooling, we will initialize the random index part.
  if (this->layer_param_.pooling_param().pool() ==
      PoolingParameter_PoolMethod_STOCHASTIC) {
    rand_idx_.Reshape(std::vector<int>{bottom[0]->shape(0), channels_, pooled_height_,
      pooled_width_, 16});
  }

  bottom_spatial_dim_ = bottom[0]->shape(2) * bottom[0]->shape(3) * bottom[0]->shape(4);
  top_spatial_dim_ = top[0]->shape(2) * top[0]->shape(3) * top[0]->shape(4);
  if(!global_pooling_) {
    if(kernel_h_ == bottom[0]->shape(2) && kernel_w_ == bottom[0]->shape(3) && 
        pad_h_ == 0 && pad_w_ == 0 && stride_h_ == 1 && stride_w_ == 1) { //then it is actually global_pooling_ 
      global_pooling_ = true;
    }
  }

#ifdef USE_AICORE
  // bottom_fp16_.Reshape(bottom[0]->shape());
  // top_fp16_.Reshape(top[0]->shape());
#endif
}

// TODO(Yangqing): Is there a faster way to do pooling in the channel-first
// case?
template <typename Dtype>
void PoolingLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const int top_count = top[0]->count();
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  int* mask = NULL;  // suppress warnings about uninitialized variables
  Dtype* top_mask = NULL;
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more code.
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // Initialize
    if (use_top_mask) {
      top_mask = top[1]->mutable_cpu_data();
      caffe_set(top_count, Dtype(-1), top_mask);
    } else {
      mask = max_idx_.mutable_cpu_data();
      caffe_set(top_count, -1, mask);
    }

    if(global_pooling_) {
      // caffe_set(top_count, Dtype(-FLT_MAX), top_data);
      // // The main loop
      // for (int n = 0; n < bottom[0]->shape(0); ++n) {
      //   for (int c = 0; c < channels_; ++c) {
      //     bottom_data = bottom[0]->cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;
      //     top_data = top[0]->mutable_cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;top[0]->shape(2) * top[0]->shape(3) * 16;
      //     if (use_top_mask) {
      //       top_mask = top[1]->mutable_cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;top[0]->shape(2) * top[0]->shape(3) * 16;
      //     } else {
      //       mask = max_idx_.mutable_cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;top[0]->shape(2) * top[0]->shape(3) * 16;
      //     }
      //     for (int ph = 0; ph < pooled_height_; ++ph) {
      //       for (int pw = 0; pw < pooled_width_; ++pw) {
      //         int hstart = ph * stride_h_ - pad_h_;
      //         int wstart = pw * stride_w_ - pad_w_;
      //         int hend = min(hstart + kernel_h_, height_);
      //         int wend = min(wstart + kernel_w_, width_);
      //         hstart = max(hstart, 0);
      //         wstart = max(wstart, 0);
      //         int pool_index = ph * pooled_width_ + pw;
      //         for (int h = hstart; h < hend; ++h) {
      //             for (int w = wstart; w < wend; ++w) {
      //               for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
      //               const int index = (h * width_ + w) * CHANNEL_16 + c0;
      //               if (bottom_data[index] > top_data[pool_index * CHANNEL_16 + c0]) {
      //                 top_data[pool_index * CHANNEL_16 + c0] = bottom_data[index];
      //                 if (use_top_mask) {
      //                   top_mask[pool_index * CHANNEL_16 + c0] = static_cast<Dtype>(index);
      //                 } else {
      //                   mask[pool_index * CHANNEL_16 + c0] = index;
      //                 }
      //               }
      //             }
      //           }
      //         }
      //       }
      //     }
      //   }
      // }
    } else {
      caffe_set(top_count, Dtype(-FLT_MAX), top_data);
      // The main loop
      
      if(use_top_mask) {
        NOT_IMPLEMENTED;
      }
      if(pad_h_ == 0 && pad_w_ == 0) {
        #pragma omp parallel for
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < channels_; ++c) {
            bottom_data = bottom[0]->cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;
            top_data = top[0]->mutable_cpu_data() + (n * channels_ + c) * top_spatial_dim_;
            mask = max_idx_.mutable_cpu_data() + (n * channels_ + c) * top_spatial_dim_;
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int pool_index = ph * pooled_width_ + pw * CHANNEL_16;
                for (int h = ph * stride_h_; h < ph * stride_h_+ kernel_h_; ++h) {
                    for (int w = pw * stride_w_; w < pw * stride_w_ + kernel_w_; ++w) {
                      const int index = (h * width_ + w) * CHANNEL_16;
                      for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
                      if (bottom_data[index+c0] > top_data[pool_index + c0]) {
                        top_data[pool_index + c0] = bottom_data[index+c0];
                        mask[pool_index + c0] = index+c0;
                      }
                    }
                  }
                }
              }
            }
          }
        }        
      } else {
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < channels_; ++c) {
            bottom_data = bottom[0]->cpu_data() + (n * channels_ + c) * bottom[0]->shape(2) * bottom[0]->shape(3) * 16;
            top_data = top[0]->mutable_cpu_data() + (n * channels_ + c) * top[0]->shape(2) * top[0]->shape(3) * 16;
            if (use_top_mask) {
              top_mask = top[1]->mutable_cpu_data() + (n * channels_ + c) * top[0]->shape(2) * top[0]->shape(3) * 16;
            } else {
              mask = max_idx_.mutable_cpu_data() + (n * channels_ + c) * top[0]->shape(2) * top[0]->shape(3) * 16;
            }
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                int hstart = ph * stride_h_ - pad_h_;
                int wstart = pw * stride_w_ - pad_w_;
                int hend = min(hstart + kernel_h_, height_);
                int wend = min(wstart + kernel_w_, width_);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                int pool_index = ph * pooled_width_ + pw;
                for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
                      const int index = (h * width_ + w) * CHANNEL_16 + c0;
                      if (bottom_data[index] > top_data[pool_index * CHANNEL_16 + c0]) {
                        top_data[pool_index * CHANNEL_16 + c0] = bottom_data[index];
                        if (use_top_mask) {
                          top_mask[pool_index * CHANNEL_16 + c0] = static_cast<Dtype>(index);
                        } else {
                          mask[pool_index * CHANNEL_16 + c0] = index;
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:

    if(global_pooling_) {
      #pragma omp parallel for
      for (int n = 0; n < bottom[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          bottom_data = bottom[0]->cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;
          top_data = top[0]->mutable_cpu_data() + (n * channels_ + c) * top_spatial_dim_;
          for (int x = 0; x < kernel_h_ * kernel_w_; ++x) { //global pooling we are safe to do so
            for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
              top_data[c0] += bottom_data[x * CHANNEL_16 + c0];
            }
          }
          for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
            top_data[c0] /= pool_size_;
          }
        }
      }
    } else {
      #pragma omp parallel for
      for (int i = 0; i < top_count; ++i) {
        top_data[i] = 0;
      }
      // The main loop
      if(pad_h_ == 0 && pad_w_ == 0) {
        #pragma omp parallel for
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < channels_; ++c) {
            bottom_data = bottom[0]->cpu_data() + (n * channels_ + c) * bottom_spatial_dim_;
            top_data = top[0]->mutable_cpu_data() + (n * channels_ + c) * top_spatial_dim_;
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                for (int h = ph * stride_h_; h < ph * stride_h_ + kernel_h_; ++h) {
                  for (int w = pw * stride_w_; w < pw * stride_w_ + kernel_w_; ++w) {
                    for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
                      top_data[(ph * pooled_width_ + pw) * CHANNEL_16 + c0] +=
                          bottom_data[(h * width_ + w) * CHANNEL_16 + c0];
                    }
                  }
                }
                for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
                  top_data[(ph * pooled_width_ + pw) * CHANNEL_16 + c0] /= pool_size_;
                }
              }
            }
          }
        }
      } else {
        for (int n = 0; n < bottom[0]->shape(0); ++n) {
          for (int c = 0; c < channels_; ++c) {
            bottom_data = bottom[0]->cpu_data() + (n * channels_ + c) * bottom[0]->shape(2) * bottom[0]->shape(3) * 16;
            top_data = top[0]->mutable_cpu_data() + (n * channels_ + c) * top[0]->shape(2) * top[0]->shape(3) * 16;
            for (int ph = 0; ph < pooled_height_; ++ph) {
              for (int pw = 0; pw < pooled_width_; ++pw) {
                for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
                  int hstart = ph * stride_h_ - pad_h_;
                  int wstart = pw * stride_w_ - pad_w_;
                  int hend = min(hstart + kernel_h_, height_ + pad_h_);
                  int wend = min(wstart + kernel_w_, width_ + pad_w_);
                  int pool_size = (hend - hstart) * (wend - wstart);
                  hstart = max(hstart, 0);
                  wstart = max(wstart, 0);
                  hend = min(hend, height_);
                  wend = min(wend, width_);
                  for (int h = hstart; h < hend; ++h) {
                    for (int w = wstart; w < wend; ++w) {
                      top_data[(ph * pooled_width_ + pw) * CHANNEL_16 + c0] +=
                          bottom_data[(h * width_ + w) * CHANNEL_16 + c0];
                    }
                  }
                  top_data[(ph * pooled_width_ + pw) * CHANNEL_16 + c0] /= pool_size;
                }
              }
            }
          }
        }
      }
    }
  
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  if (!propagate_down[0]) {
    return;
  }
  const Dtype* top_diff = top[0]->cpu_diff();
  Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  // Different pooling methods. We explicitly do the switch outside the for
  // loop to save time, although this results in more codes.
  caffe_set(bottom[0]->count(), Dtype(0), bottom_diff);
  // We'll output the mask to top[1] if it's of size >1.
  const bool use_top_mask = top.size() > 1;
  const int* mask = NULL;  // suppress warnings about uninitialized variables
  const Dtype* top_mask = NULL;
  switch (this->layer_param_.pooling_param().pool()) {
  case PoolingParameter_PoolMethod_MAX:
    // The main loop
    if (use_top_mask) {
      NOT_IMPLEMENTED;
      top_mask = top[1]->cpu_data();
    } else {
      mask = max_idx_.cpu_data();
    }
    #pragma omp parallel for
    for (int n = 0; n < top[0]->shape(0); ++n) {
      for (int c = 0; c < channels_; ++c) {
        bottom_diff = bottom[0]->mutable_cpu_diff() + (n * channels_ + c) * bottom_spatial_dim_;
        top_diff = top[0]->cpu_diff() + (n * channels_ + c) * top_spatial_dim_;
        mask = max_idx_.cpu_data() + (n * channels_ + c) * top_spatial_dim_;
        for (int ph = 0; ph < pooled_height_; ++ph) {
          for (int pw = 0; pw < pooled_width_; ++pw) {
            for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
              const int index = (ph * pooled_width_ + pw) * CHANNEL_16 + c0;
              bottom_diff[mask[index]] += top_diff[index];
            }
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_AVE:
    // The main loop
    if(global_pooling_) {
      #pragma omp parallel for
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          bottom_diff = bottom[0]->mutable_cpu_diff() + (n * channels_ + c) * bottom_spatial_dim_;
          top_diff = top[0]->cpu_diff() + (n * channels_ + c) * top_spatial_dim_;
          for (int x = 0; x < kernel_h_ * kernel_w_; ++x) { //global pooling we are safe to do so
            for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
              bottom_diff[x * CHANNEL_16 + c0] += top_diff[c0] / pool_size_;
            }
          }
        }
      }
    } else {
      for (int n = 0; n < top[0]->shape(0); ++n) {
        for (int c = 0; c < channels_; ++c) {
          bottom_diff = bottom[0]->mutable_cpu_diff() + (n * channels_ + c) * bottom[0]->shape(2) * bottom[0]->shape(3) * 16;
          top_diff = top[0]->cpu_diff() + (n * channels_ + c) * top[0]->shape(2) * top[0]->shape(3) * 16;
          for (int ph = 0; ph < pooled_height_; ++ph) {
            for (int pw = 0; pw < pooled_width_; ++pw) {
              for (int c0 = 0; c0 < CHANNEL_16; ++c0) {
                int hstart = ph * stride_h_ - pad_h_;
                int wstart = pw * stride_w_ - pad_w_;
                int hend = min(hstart + kernel_h_, height_ + pad_h_);
                int wend = min(wstart + kernel_w_, width_ + pad_w_);
                int pool_size = (hend - hstart) * (wend - wstart);
                hstart = max(hstart, 0);
                wstart = max(wstart, 0);
                hend = min(hend, height_);
                wend = min(wend, width_);
                for (int h = hstart; h < hend; ++h) {
                  for (int w = wstart; w < wend; ++w) {
                    bottom_diff[(h * width_ + w) * CHANNEL_16 + c0] +=
                      top_diff[(ph * pooled_width_ + pw) * CHANNEL_16 + c0] / pool_size;
                  }
                }
              }
            }
          }
        }
      }
    }
    break;
  case PoolingParameter_PoolMethod_STOCHASTIC:
    NOT_IMPLEMENTED;
    break;
  default:
    LOG(FATAL) << "Unknown pooling method.";
  }
}


#ifdef CPU_ONLY
STUB_GPU(PoolingLayer);
#endif

INSTANTIATE_CLASS(PoolingLayer);

}  // namespace caffe
