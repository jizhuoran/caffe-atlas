#include <algorithm>
#include <string>
#include <vector>
#include "caffe/common.hpp"
#include "caffe/layers/softmax_layer.hpp"

namespace caffe {

template <typename Dtype>
void SoftmaxLayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  auto hack_str = new std::string(fmt::format("softmax_fw_{}_{}__kernel0", outer_num_, bottom[0]->count() / outer_num_));
  auto err = custom::op_run(*hack_str, 
                                   0,
                                   fmt::format("{}/softmax_fw_{}_{}.o", Caffe::kernel_dir(), outer_num_, bottom[0]->count() / outer_num_),
                                   {bottom[0]->aicore_data()},
                                   {top[0]->mutable_aicore_data()},
                                   {top[0]->count() * static_cast<unsigned int>(sizeof(half))},
                                   1,
                                   {128, 128, 1280, 1280});

  AICORE_CHECK(err);

}
 
template <typename Dtype>
void SoftmaxLayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

        Backward_cpu(top, propagate_down, bottom);
//   if (propagate_down[0]) {
//     auto hack_str = new std::string(fmt::format("ReLU_bw_{}__kernel0", bottom[0]->count()));
//     auto err = custom::op_run(*hack_str, 
//                                    0,
//                                    fmt::format("{}/ReLU_bw_{}.o", Caffe::kernel_dir(), bottom[0]->count()),
//                                    {bottom[0]->aicore_data(), top[0]->aicore_diff()},
//                                    {bottom[0]->mutable_aicore_diff()},
//                                    {bottom[0]->count() * static_cast<unsigned int>(sizeof(half))});
//     AICORE_CHECK(err); 
//   }                  
}

INSTANTIATE_LAYER_AICORE_FUNCS(SoftmaxLayer);

}  // namespace caffe
