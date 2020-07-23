#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include "caffe/blob.hpp"
#include "caffe/common.hpp"
#include "caffe/layers/inner_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

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

    // if(top[0]->shape(1) < 32) {
    //     Forward_cpu(bottom, top);
    //     return;
    // }

    auto aligned_weight = this->blobs_[0].get();
    //align_mm(this->blobs_[0]->cpu_data(), aligned_weight.mutable_cpu_data(), this->blobs_[0]->shape(0), this->blobs_[0]->shape(1));
    Blob<Dtype> aligned_bottom(std::vector<int>{ALIGN_SIZE(bottom[0]->shape(0)), ALIGN_SIZE(K_)});
    align_mm(bottom[0]->cpu_data(), aligned_bottom.mutable_cpu_data(), bottom[0]->shape(0), K_);

    Blob<Dtype> aligned_top(std::vector<int>{ALIGN_SIZE(top[0]->shape(0)), ALIGN_SIZE(top[0]->shape(1))});

    // auto hack_str = new std::string(fmt::format("matmul_op_{}_{}_{}_{}_{}_{}__kernel0", M_, ALIGN_SIZE(K_),
    //                                    ALIGN_SIZE(N_), "NTA", 
    //                                    transpose_? "NTB" : "TB",
    //                                    (bias_term_ ? "bias" : "nobias")));


    // std::cout << "fw: " << *hack_str << std::endl;
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

    AICORE_CHECK(rtKernelLaunch(this->fw_kernel, this->fw_block_num, args.data(), args.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
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

            AICORE_CHECK(rtKernelLaunch(this->bw_weight_kernel, this->bw_weight_block_num, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
            AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));




        } else {

            std::vector<void*> args1 = { (void*)aligned_top.aicore_diff(),
                                         (void*)aligned_bottom.aicore_data(),
                                         (void*)aligned_weight->mutable_aicore_diff()};

            AICORE_CHECK(rtKernelLaunch(this->bw_weight_kernel, this->bw_weight_block_num, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
            AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

            auto temp = aligned_weight->cpu_data();
            for(int i = 0; i < aligned_weight->count(); ++i) {
                std::cout << temp[i] << " ";
            }
            std::cout << " " << std::endl;
        }


    }





    if (bias_term_ && this->param_propagate_down_[1]) {
        const Dtype* top_diff = top[0]->cpu_diff();
        Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();

        for(int n = 0; n < N_; ++n) {
            for(int m = 0; m < M_; ++m) {
                // bias_diff[0] += top_diff[0];
                bias_diff[n] += top_diff[n * M_ + m];
            }
        }
        // Gradient with respect to bias
        // caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        //     bias_multiplier_.cpu_data(), (Dtype)1.,
        //     this->blobs_[1]->mutable_cpu_diff());
    }


    if (propagate_down[0]) {

        std::vector<void*> args1 = { (void*)aligned_top.aicore_diff(),
                                        (void*)aligned_weight->aicore_data(),
                                        (void*)aligned_bottom.mutable_aicore_diff()};

        AICORE_CHECK(rtKernelLaunch(this->bw_input_kernel, this->bw_input_block_num, args1.data(), args1.size() * sizeof(void*), NULL, Caffe::Get().aicore_stream));
        AICORE_CHECK(rtStreamSynchronize(Caffe::Get().aicore_stream));

        // vector<Dtype> tttt(aligned_bottom.count(), Dtype(.0));

        // AICORE_CHECK(rtMemcpy(tttt.data(), aligned_bottom.count() * sizeof(Dtype), (void *)aligned_bottom.aicore_diff(), aligned_bottom.count() * sizeof(Dtype), RT_MEMCPY_DEVICE_TO_HOST));


        Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
        const Dtype* aligned_bottom_diff = aligned_bottom.aicore_diff();

        for(int i = 0; i < bottom[0]->shape(0); ++i) {
            caffe_copy(bottom[0]->shape(1), aligned_bottom_diff + i * aligned_bottom.shape(1), bottom_diff + i * K_);
        }
    }               
}

INSTANTIATE_LAYER_AICORE_FUNCS(InnerProductLayer);

}  // namespace caffe
