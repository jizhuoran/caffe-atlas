#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  auto hack_str = new std::string(fmt::format("ReLU_fw_{}__kernel0", bottom[0]->count()));
  auto err = custom::op_run(*hack_str, 
                                   0,
                                   fmt::format("{}/ReLU_fw_{}.o", Caffe::kernel_dir(), bottom[0]->count()),
                                   {bottom[0]->aicore_data()},
                                   {top[0]->mutable_aicore_data()},
                                   {bottom[0]->count() * static_cast<unsigned int>(sizeof(half))},
                                   2);

  AICORE_CHECK(err);
}
 
template <typename Dtype>
void ReLULayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (propagate_down[0]) {
    auto hack_str = new std::string(fmt::format("ReLU_bw_{}__kernel0", bottom[0]->count()));
    auto err = custom::op_run(*hack_str, 
                                   0,
                                   fmt::format("{}/ReLU_bw_{}.o", Caffe::kernel_dir(), bottom[0]->count()),
                                   {bottom[0]->aicore_data(), top[0]->aicore_diff()},
                                   {bottom[0]->mutable_aicore_diff()},
                                   {bottom[0]->count() * static_cast<unsigned int>(sizeof(half))},
                                   2);
    AICORE_CHECK(err); 
  }                  
}

INSTANTIATE_LAYER_AICORE_FUNCS(ReLULayer);

}  // namespace caffe
