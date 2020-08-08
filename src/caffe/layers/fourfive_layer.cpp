#include <vector>

#include "caffe/aicore_layers/fourfive_layer.hpp"

namespace caffe {

template <typename Dtype>
void FourFiveLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  CHECK_NE(top[0], bottom[0]) << this->type() << " Layer does not "
      "allow in-place computation.";

  if(this->layer_param_.fourfive_param().four2five()) {
    four2five_ = true;
    CHECK_EQ(bottom[0]->shape().size(), 4);
    vector<int> top_shape {bottom[0]->shape(0), (bottom[0]->shape(1) + 15)/16, bottom[0]->shape(2), bottom[0]->shape(3), 16};
    top[0]->Reshape(top_shape);
  } else {
    four2five_ = false;
    CHECK_EQ(bottom[0]->shape().size(), 5);
    channels_ = this->layer_param_.fourfive_param().channels();
    vector<int> top_shape {bottom[0]->shape(0), channels_, bottom[0]->shape(2), bottom[0]->shape(3)};
    top[0]->Reshape(top_shape);
  }


}

template <typename Dtype>
void FourFiveLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  if(four2five_) {
    // if(bottom[0]->fp16_copy_ != nullptr && top[0]->fp16_copy_ != nullptr) {
    //   LOG(ERROR) << "Interesting, why both are fp16 and fourfive conversion is still needed";
    // } else if (bottom[0]->fp16_copy_ != nullptr) {
    //   four2five(bottom[0]->fp16_copy_->cpu_data(), top[0]->mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));
    // } else if (top[0]->fp16_copy_ != nullptr) {
    //   four2five(bottom[0]->cpu_data(), top[0]->fp16_copy_->mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));
    // } else {
      four2five(bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));
    // }  
  } else {
    // if(bottom[0]->fp16_copy_ != nullptr && top[0]->fp16_copy_ != nullptr) {
    //   LOG(ERROR) << "Interesting, why both are fp16 and fourfive conversion is still needed";
    // } else if (bottom[0]->fp16_copy_ != nullptr) {
    //   five2four(bottom[0]->fp16_copy_->cpu_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
    // } else if (top[0]->fp16_copy_ != nullptr) {
    //   five2four(bottom[0]->cpu_data(), top[0]->fp16_copy_->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
    // } else {
      five2four(bottom[0]->cpu_data(), top[0]->mutable_cpu_data(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
    // }  
  }
}

template <typename Dtype>
void FourFiveLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {

  if(four2five_) {
    five2four(top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(), bottom[0]->shape(0), bottom[0]->shape(1), bottom[0]->shape(2), bottom[0]->shape(3));
  } else {
    four2five(top[0]->cpu_diff(), bottom[0]->mutable_cpu_diff(), top[0]->shape(0), top[0]->shape(1), top[0]->shape(2), top[0]->shape(3));
  }
}

INSTANTIATE_CLASS(FourFiveLayer);
REGISTER_LAYER_CLASS(FourFive);

}  // namespace caffe
