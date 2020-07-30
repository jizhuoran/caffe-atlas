#include "caffe/layers/relu_layer.hpp"

namespace caffe {

template <typename Dtype>
void ReLULayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

#ifdef USE_AICORE
  if(Caffe::aicore_mode()) {
    AicoreKerel fw_param = this->layer_param_.aicorekernel(0);
    char* fw_stub = Caffe::Get().new_load_aicore_kernel(fw_param.kernelfile(), fw_param.kernelname());
    this->aicore_kernel_info_.push_back(AICoreKernelInfo(fw_stub, fw_param.block_num()));

    AicoreKerel bw_input_param = this->layer_param_.aicorekernel(1);
    char* bw_input_stub = Caffe::Get().new_load_aicore_kernel(bw_input_param.kernelfile(), bw_input_param.kernelname());
    this->aicore_kernel_info_.push_back(AICoreKernelInfo(bw_input_stub, bw_input_param.block_num()));
  }
#endif
}

template
void ReLULayer<float>::LayerSetUp(const vector<Blob<float>*>& bottom, const vector<Blob<float>*>& top);

template
void ReLULayer<double>::LayerSetUp(const vector<Blob<double>*>& bottom, const vector<Blob<double>*>& top);



template <typename Dtype>
void ReLULayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  
  // Forward_cpu(bottom, top);
  // return;
  std::vector<void*> args = {(void*)bottom[0]->aicore_data(), (void*)top[0]->mutable_aicore_data()};

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[0].kernel_, this->aicore_kernel_info_[0].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

}
 
template <typename Dtype>
void ReLULayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  // Backward_cpu(top, propagate_down, bottom);
  // return;

   std::vector<void*> args = {(void*)top[0]->aicore_diff(), (void*)bottom[0]->aicore_data(), (void*)bottom[0]->mutable_aicore_diff()};

  AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[1].kernel_, this->aicore_kernel_info_[1].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
  AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));                  
}

INSTANTIATE_LAYER_AICORE_FUNCS(ReLULayer);

}  // namespace caffe
