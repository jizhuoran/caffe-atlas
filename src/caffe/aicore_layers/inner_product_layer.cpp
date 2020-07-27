#include "caffe/layers/inner_product_layer.hpp"

namespace caffe {

template <typename Dtype>
void align_mm(const Dtype *in, Dtype *out, int M, int N) {
    int aligned_N = ALIGN_SIZE(N);
    if(aligned_N == N) {
        caffe_copy(M*N, in, out);
    } else {
        for(int m = 0; m < M; ++m) {
            caffe_copy(N, in + m * N, out + m * aligned_N);
        }
    }

}

template <typename Dtype>
void InnerProductLayer<Dtype>::Forward_aicore(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

    auto aligned_weight = this->blobs_[0].get();
    Blob<Dtype> aligned_bottom(std::vector<int>{ALIGN_SIZE(bottom[0]->shape(0)), ALIGN_SIZE(K_)});
    align_mm(bottom[0]->cpu_data(), aligned_bottom.mutable_cpu_data(), bottom[0]->shape(0), K_);

    Blob<Dtype> aligned_top(std::vector<int>{ALIGN_SIZE(top[0]->shape(0)), ALIGN_SIZE(top[0]->shape(1))});

    std::unique_ptr<Blob<Dtype>> aligned_bias;

    if(bias_term_) {
        vector<int> bias_shape{top[0]->shape(0), ALIGN_SIZE(N_)};
        aligned_bias.reset(new Blob<Dtype>(bias_shape));
        auto bias_data = this->blobs_[1]->cpu_data();
        auto algin_bias_data = aligned_bias->mutable_cpu_data();
        for(int i = 0; i < top[0]->shape(0); ++i) {
            caffe_copy(N_, bias_data, algin_bias_data+ i * aligned_bias->shape(1));
        }
        align_mm(this->blobs_[1]->cpu_data(), aligned_bias->mutable_cpu_data(), top[0]->shape(0), N_);
    }



    std::vector<void*> args = {(void*)aligned_bottom.aicore_data(), (void*)aligned_weight->aicore_data()};
    if (this->bias_term_) {
        args.push_back((void*)aligned_bias->aicore_data());
    }
    args.push_back((void*)aligned_top.mutable_aicore_data());

    AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[0].kernel_, this->aicore_kernel_info_[0].block_num_, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
    AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

    Dtype* top_data = top[0]->mutable_cpu_data();
    const Dtype* aligned_top_data = aligned_top.aicore_data();
    for(int i = 0; i < top[0]->shape(0); ++i) {
        caffe_copy(top[0]->shape(1), aligned_top_data + i * aligned_top.shape(1), top_data+ i * top[0]->shape(1));
    }

}

template <typename Dtype>
void InnerProductLayer<Dtype>::Backward_aicore(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
    
    Blob<Dtype> aligned_bottom(std::vector<int>{ALIGN_SIZE(bottom[0]->shape(0)), ALIGN_SIZE(K_)});
    align_mm(bottom[0]->cpu_data(), aligned_bottom.mutable_cpu_data(), bottom[0]->shape(0), K_);

    Blob<Dtype> aligned_top(std::vector<int>{ALIGN_SIZE(top[0]->shape(0)), ALIGN_SIZE(top[0]->shape(1))});
    align_mm(top[0]->cpu_diff(), aligned_top.mutable_cpu_diff(), top[0]->shape(0), top[0]->shape(1));
       
    auto aligned_weight = this->blobs_[0].get();

    if (this->param_propagate_down_[0]) {
        
        if (transpose_) {


            std::vector<void*> args1 = { (void*)aligned_bottom.aicore_data(), 
                              (void*)aligned_top.aicore_diff(),
                              (void*)aligned_weight->mutable_aicore_diff()};

            AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[1].kernel_, this->aicore_kernel_info_[1].block_num_, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
            AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));




        } else {

            std::vector<void*> args1 = { (void*)aligned_top.aicore_diff(),
                                         (void*)aligned_bottom.aicore_data(),
                                         (void*)aligned_weight->mutable_aicore_diff()};

            AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[1].kernel_, this->aicore_kernel_info_[1].block_num_, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
            AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

        }


    }


    if (bias_term_ && this->param_propagate_down_[1]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();

        for(int n = 0; n < N_; ++n) {
            for(int m = 0; m < M_; ++m) {
                bias_diff[n] += top_diff[n * M_ + m];
            }
        }
    }


    if (propagate_down[0]) {

        std::vector<void*> args1 = { (void*)aligned_top.aicore_diff(),
                                        (void*)aligned_weight->aicore_data(),
                                        (void*)aligned_bottom.mutable_aicore_diff()};

        AICORE_CHECK(rtKernelLaunch(this->aicore_kernel_info_[2].kernel_, this->aicore_kernel_info_[2].block_num_, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
        AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* aligned_bottom_diff = aligned_bottom.aicore_diff();

        for(int i = 0; i < bottom[0]->shape(0); ++i) {
            caffe_copy(bottom[0]->shape(1), aligned_bottom_diff + i * aligned_bottom.shape(1), bottom_diff + i * K_);
        }
    }               
}

INSTANTIATE_LAYER_AICORE_FUNCS(InnerProductLayer);

}  // namespace caffe
