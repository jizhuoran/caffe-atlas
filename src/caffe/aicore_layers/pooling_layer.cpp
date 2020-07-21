#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/layers/pooling_layer.hpp"

namespace caffe {

template <typename Dtype>
void PoolingLayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  Forward_cpu(bottom, top);
  // return; //Pooling has too many bugs

  // std::string pooling_mode;
  // if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_MAX) {
  //   pooling_mode = "MAX";
  // } else if (this->layer_param_.pooling_param().pool() == PoolingParameter_PoolMethod_AVE) {
  //   pooling_mode = "AVG";
  // }

  // Blob<Dtype> bottom_five(bottom[0]->shape(0), (bottom[0]->shape(1)+15)/16*16, bottom[0]->shape(2), bottom[0]->shape(3));
  // four2five(bottom[0]->cpu_data(), bottom_five.mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));


  // std::vector<int> top_five_shape{top[0]->shape(0), (top[0]->shape(1) + 15) / 16, top[0]->shape(2), top[0]->shape(3) * 16};
  // Blob<Dtype> top_five(top_five_shape);


  // auto hack_str = new std::string(fmt::format("pooling_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}__kernel0", pooling_mode, "SAME",
  //                                      bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3), 
  //                                      kernel_h_, kernel_w_, stride_h_, stride_w_));

  // auto err = custom::op_run(*hack_str,
  //                                  0,
  //                                  fmt::format("{}/pooling_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.o", Caffe::kernel_dir(), pooling_mode, "SAME",
  //                                      bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3), 
  //                                      kernel_h_, kernel_w_, stride_h_, stride_w_),
  //                                  {bottom_five.1aicore_data()},
  //                                  {top_five.1mutable_aicore_data()},
  //                                  {top_five.count() * static_cast<unsigned int>(sizeof(half))});

  // AICORE_EXEC_CHECK(err);
  // five2four(top_five.cpu_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));

}

template <typename Dtype>
void PoolingLayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  Backward_cpu(top, propagate_down, bottom);
                
}

INSTANTIATE_LAYER_AICORE_FUNCS(PoolingLayer);

}  // namespace caffe
