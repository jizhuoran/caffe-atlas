#include <boost/math/special_functions/next.hpp>
#include <boost/random.hpp>

#include <ios>
#include <limits>
#include <random>

#include "caffe/common.hpp"
#include "caffe/util/half.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/armblas_fp16.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
void caffe_aicore_set(const int N, const Dtype alpha, std::string Y) {
  std::vector<_Float16> data(N, alpha);
  caffe_aicore_memcpy(N * sizeof(_Float16), data.data(), Y); //FIXME
}

template <typename Dtype>
void caffe_aicore_set(const int N, const Dtype alpha, Dtype* Y) {
  AICORE_CHECK(rtMemset(Y, N * sizeof(Dtype), alpha , N));
}

template void caffe_aicore_set<int>(const int N, const int alpha, std::string Y);
template void caffe_aicore_set<float>(const int N, const float alpha, std::string Y);
template void caffe_aicore_set<double>(const int N, const double alpha, std::string Y);
template void caffe_aicore_set<_Float16>(const int N, const _Float16 alpha, std::string Y);

template void caffe_aicore_set<int>(const int N, const int alpha, int* Y);
template void caffe_aicore_set<float>(const int N, const float alpha, float* Y);
template void caffe_aicore_set<double>(const int N, const double alpha, double* Y);
template void caffe_aicore_set<_Float16>(const int N, const _Float16 alpha, _Float16* Y);

// void caffe_aicore_memcpy(const size_t N, const void *X, void *Y){
//   if (X != Y) {
//     AICORE_CHECK(cudaMemcpy(Y, X, N, cudaMemcpyDefault));  // NOLINT(caffe/alt_fn)
//   }
// }

void caffe_aicore_memcpy(const size_t N, std::string X, void *Y) {
  std::ifstream f(X, std::ios::binary);
  f.read(reinterpret_cast<char*>(Y), N);
  f.close();
}
void caffe_aicore_memcpy(const size_t N, const void *X, std::string Y) {
  std::ofstream f(Y, std::ios::binary | std::ios::trunc);
  f.write(reinterpret_cast<const char*>(X), N);
  f.close();
}

void caffe_aicore_memset(const size_t N, const char alpha, std::string X) {
  std::vector<char> data(N, alpha);
  std::ofstream f(X, std::ios::binary | std::ios::trunc);
  f.write(data.data(), N);
  f.close();
}

void caffe_aicore_memset(const size_t N, const int alpha, void* X) {
  AICORE_CHECK(rtMemset(X, N, alpha, N));
}

template <typename Dtype>
void five2four(const Dtype* five, Dtype* four, int batch_size, int channel_in, int in_height, int in_width) {
  auto five_array = *reinterpret_cast<const Dtype (*)[batch_size][(channel_in+15)/16][in_height][in_width][16]>(five);
  auto four_array = *reinterpret_cast<Dtype (*)[batch_size][channel_in][in_height][in_width]>(four);
  
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
template void five2four<float>(const float* five, float* four, int batch_size, int channel_in, int in_height, int in_width);
template void five2four<double>(const double* five, double* four, int batch_size, int channel_in, int in_height, int in_width);
template void five2four<_Float16>(const _Float16* five, _Float16* four, int batch_size, int channel_in, int in_height, int in_width);


template <typename Dtype>
void four2five(const Dtype* four, Dtype* five, int batch_size, int channel_in, int in_height, int in_width) {
  auto five_array = *reinterpret_cast<Dtype (*)[batch_size][(channel_in+15)/16][in_height][in_width][16]>(five);
  auto four_array = *reinterpret_cast<const Dtype (*)[batch_size][channel_in][in_height][in_width]>(four);
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
template void four2five<float>(const float* four, float* five, int batch_size, int channel_in, int in_height, int in_width);
template void four2five<double>(const double* four, double* five, int batch_size, int channel_in, int in_height, int in_width);
template void four2five<_Float16>(const _Float16* four, _Float16* five, int batch_size, int channel_in, int in_height, int in_width);





template<>
void caffe_cpu_gemm<float>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<double>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C) {
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
      ldb, beta, C, N);
}

template<>
void caffe_cpu_gemm<_Float16>(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const _Float16 alpha, const _Float16* A, const _Float16* B, const _Float16 beta,
    _Float16* C) {
  std::cout << "FOR Debug Only!" << std::endl;

  std::vector<float> A32(M * K);
  std::vector<float> B32(K * N);
  std::vector<float> C32(M * N);
  for(int i = 0; i < A32.size(); ++i) {
    A32[i] = float(A[i]);
  }
  for(int i = 0; i < B32.size(); ++i) {
    B32[i] = float(B[i]);
  }
  for(int i = 0; i < C32.size(); ++i) {
    C32[i] = float(C[i]);
  }
  caffe_cpu_gemm<float>(TransA, TransB, M, N, K, float(alpha), A32.data(), B32.data(), float(beta), C32.data());
  for(int i = 0; i < C32.size(); ++i) {
    C[i] = _Float16(C32[i]);
  }
}

template <>
void caffe_cpu_gemv<float>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y) {
  cblas_sgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<double>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y) {
  cblas_dgemv(CblasRowMajor, TransA, M, N, alpha, A, N, x, 1, beta, y, 1);
}

template <>
void caffe_cpu_gemv<_Float16>(const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const _Float16 alpha, const _Float16* A, const _Float16* x,
    const _Float16 beta, _Float16* y) {
  std::cout << "FOR Debug Only!" << std::endl;

  std::vector<float> A32(M * N);
  std::vector<float> x32((TransA == CblasNoTrans)? N:M);
  std::vector<float> y32((TransA == CblasNoTrans)? M:N);
  for(int i = 0; i < A32.size(); ++i) {
    A32[i] = float(A[i]);
  }
  for(int i = 0; i < x32.size(); ++i) {
    x32[i] = float(x[i]);
  }
  for(int i = 0; i < y32.size(); ++i) {
    y32[i] = float(y[i]);
  }
  caffe_cpu_gemv<float>(TransA, M, N, float(alpha), A32.data(), x32.data(), float(beta), y32.data());
  for(int i = 0; i < y32.size(); ++i) {
    y[i] = _Float16(y32[i]);
  }
}

template <>
void caffe_axpy<float>(const int N, const float alpha, const float* X,
    float* Y) { cblas_saxpy(N, alpha, X, 1, Y, 1); }

template <>
void caffe_axpy<double>(const int N, const double alpha, const double* X,
    double* Y) { cblas_daxpy(N, alpha, X, 1, Y, 1); }
  
template <>
void caffe_axpy<_Float16>(const int N, const _Float16 alpha, const _Float16* X,
    _Float16* Y) { cblas_haxpy(N, alpha, X, 1, Y, 1); }

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype* Y) {
  if (alpha == 0) {
    memset(Y, 0, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    return;
  }
  for (int i = 0; i < N; ++i) {
    Y[i] = alpha;
  }
}

template void caffe_set<int>(const int N, const int alpha, int* Y);
template void caffe_set<float>(const int N, const float alpha, float* Y);
template void caffe_set<double>(const int N, const double alpha, double* Y);
template void caffe_set<_Float16>(const int N, const _Float16 alpha, _Float16* Y);

template <>
void caffe_add_scalar(const int N, const float alpha, float* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const double alpha, double* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <>
void caffe_add_scalar(const int N, const _Float16 alpha, _Float16* Y) {
  for (int i = 0; i < N; ++i) {
    Y[i] += alpha;
  }
}

template <typename Dtype>
void caffe_copy(const int N, const Dtype* X, Dtype* Y) {
  if (X != Y) {
    if (Caffe::mode() == Caffe::GPU) {
#ifndef CPU_ONLY
      // NOLINT_NEXT_LINE(caffe/alt_fn)
      CUDA_CHECK(cudaMemcpy(Y, X, sizeof(Dtype) * N, cudaMemcpyDefault));
#else
      NO_GPU;
#endif
    } else {
      memcpy(Y, X, sizeof(Dtype) * N);  // NOLINT(caffe/alt_fn)
    }
  }
}

template void caffe_copy<int>(const int N, const int* X, int* Y);
template void caffe_copy<unsigned int>(const int N, const unsigned int* X,
    unsigned int* Y);
template void caffe_copy<float>(const int N, const float* X, float* Y);
template void caffe_copy<double>(const int N, const double* X, double* Y);
template void caffe_copy<_Float16>(const int N, const _Float16* X, _Float16* Y);

template <>
void caffe_scal<float>(const int N, const float alpha, float *X) {
  cblas_sscal(N, alpha, X, 1);
}

template <>
void caffe_scal<double>(const int N, const double alpha, double *X) {
  cblas_dscal(N, alpha, X, 1);
}

template <>
void caffe_scal<_Float16>(const int N, const _Float16 alpha, _Float16 *X) {
  cblas_hscal(N, alpha, X, 1);
}

template <>
void caffe_cpu_axpby<float>(const int N, const float alpha, const float* X,
                            const float beta, float* Y) {
  cblas_saxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<double>(const int N, const double alpha, const double* X,
                             const double beta, double* Y) {
  cblas_daxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_cpu_axpby<_Float16>(const int N, const _Float16 alpha, const _Float16* X,
                             const _Float16 beta, _Float16* Y) {

  std::cout << "FOR Debug Only!" << std::endl;

  // cblas_hscale(n, alpha, x, y);
  std::vector<float> x32(N);
  std::vector<float> y32(N);

  for(int i = 0; i < N; ++i) {
    x32[i] = float(X[i]);
    y32[i] = float(Y[i]);
  }
  cblas_saxpby(N, float(alpha), x32.data(), 1, float(beta), y32.data(), 1);
  for(int i = 0; i < N; ++i) {
    Y[i] = _Float16(y32[i]);
  }


  // cblas_haxpby(N, alpha, X, 1, beta, Y, 1);
}

template <>
void caffe_add<float>(const int n, const float* a, const float* b,
    float* y) {
  vsAdd(n, a, b, y);
}

template <>
void caffe_add<double>(const int n, const double* a, const double* b,
    double* y) {
  vdAdd(n, a, b, y);
}


template <>
void caffe_add<_Float16>(const int n, const _Float16* a, const _Float16* b,
    _Float16* y) {
  vhAdd(n, a, b, y);
}

template <>
void caffe_sub<float>(const int n, const float* a, const float* b,
    float* y) {
  vsSub(n, a, b, y);
}

template <>
void caffe_sub<double>(const int n, const double* a, const double* b,
    double* y) {
  vdSub(n, a, b, y);
}

template <>
void caffe_sub<_Float16>(const int n, const _Float16* a, const _Float16* b,
    _Float16* y) {
  vhSub(n, a, b, y);
}

template <>
void caffe_mul<float>(const int n, const float* a, const float* b,
    float* y) {
  vsMul(n, a, b, y);
}

template <>
void caffe_mul<double>(const int n, const double* a, const double* b,
    double* y) {
  vdMul(n, a, b, y);
}

template <>
void caffe_mul<_Float16>(const int n, const _Float16* a, const _Float16* b,
    _Float16* y) {
  vhMul(n, a, b, y);
}

template <>
void caffe_div<float>(const int n, const float* a, const float* b,
    float* y) {
  vsDiv(n, a, b, y);
}

template <>
void caffe_div<double>(const int n, const double* a, const double* b,
    double* y) {
  vdDiv(n, a, b, y);
}

template <>
void caffe_div<_Float16>(const int n, const _Float16* a, const _Float16* b,
    _Float16* y) {
  vhDiv(n, a, b, y);
}

template <>
void caffe_powx<float>(const int n, const float* a, const float b,
    float* y) {
  vsPowx(n, a, b, y);
}

template <>
void caffe_powx<double>(const int n, const double* a, const double b,
    double* y) {
  vdPowx(n, a, b, y);
}

template <>
void caffe_powx<_Float16>(const int n, const _Float16* a, const _Float16 b,
    _Float16* y) {
  vhPowx(n, a, b, y);
}

template <>
void caffe_sqr<float>(const int n, const float* a, float* y) {
  vsSqr(n, a, y);
}

template <>
void caffe_sqr<double>(const int n, const double* a, double* y) {
  vdSqr(n, a, y);
}

template <>
void caffe_sqr<_Float16>(const int n, const _Float16* a, _Float16* y) {
  vhSqr(n, a, y);
}

template <>
void caffe_sqrt<float>(const int n, const float* a, float* y) {
  vsSqrt(n, a, y);
}

template <>
void caffe_sqrt<double>(const int n, const double* a, double* y) {
  vdSqrt(n, a, y);
}

template <>
void caffe_sqrt<_Float16>(const int n, const _Float16* a, _Float16* y) {
  vhSqrt(n, a, y);
}

template <>
void caffe_exp<float>(const int n, const float* a, float* y) {
  vsExp(n, a, y);
}

template <>
void caffe_exp<double>(const int n, const double* a, double* y) {
  vdExp(n, a, y);
}

template <>
void caffe_exp<_Float16>(const int n, const _Float16* a, _Float16* y) {
  vhExp(n, a, y);
}

template <>
void caffe_log<float>(const int n, const float* a, float* y) {
  vsLn(n, a, y);
}

template <>
void caffe_log<double>(const int n, const double* a, double* y) {
  vdLn(n, a, y);
}

template <>
void caffe_log<_Float16>(const int n, const _Float16* a, _Float16* y) {
  vhLn(n, a, y);
}

template <>
void caffe_abs<float>(const int n, const float* a, float* y) {
  vsAbs(n, a, y);
}

template <>
void caffe_abs<double>(const int n, const double* a, double* y) {
  vdAbs(n, a, y);
}

template <>
void caffe_abs<_Float16>(const int n, const _Float16* a, _Float16* y) {
  vhAbs(n, a, y);
}

unsigned int caffe_rng_rand() {
  return (*caffe_rng())();
}

template <typename Dtype>
Dtype caffe_nextafter(const Dtype b) {
  return boost::math::nextafter<Dtype>(
      b, std::numeric_limits<Dtype>::max());
}

template
float caffe_nextafter(const float b);

template
double caffe_nextafter(const double b);

template <>
_Float16 caffe_nextafter(const _Float16 b) {
  return b;
}

template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype a, const Dtype b, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(float(a), float(b));
  boost::uniform_real<Dtype> random_distribution(a, caffe_nextafter<Dtype>(b));
  boost::variate_generator<caffe::rng_t*, boost::uniform_real<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_uniform<float>(const int n, const float a, const float b,
                              float* r);

template
void caffe_rng_uniform<double>(const int n, const double a, const double b,
                               double* r);

template <>
void caffe_rng_uniform<_Float16>(const int n, const _Float16 a, const _Float16 b,
                               _Float16* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_LE(float(a), float(b));

  std::random_device rd;  //Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); //Standard mersenne_twister_engine seeded with rd()
  std::default_random_engine gen1(1996);

  std::uniform_real_distribution<float> variate_generator(float(a), caffe_nextafter<float>(float(b)));
  for (int i = 0; i < n; ++i) {
    r[i] = _Float16(variate_generator(gen1));
  }
                               
}

template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype a,
                        const Dtype sigma, Dtype* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GT(float(sigma), 0);

    std::default_random_engine gen1(1996);

  std::normal_distribution<> variate_generator(a, std::nextafter(sigma, std::numeric_limits<Dtype>::max())); //    uniform_real_distribution<float> variate_generator(float(a), caffe_nextafter<float>(float(b)));
  for (int i = 0; i < n; ++i) {
    r[i] = Dtype(variate_generator(gen1));
  }


  // boost::normal_distribution<Dtype> random_distribution(a, sigma);
  // boost::variate_generator<caffe::rng_t*, boost::normal_distribution<Dtype> >
  //     variate_generator(caffe_rng(), random_distribution);
  // for (int i = 0; i < n; ++i) {
  //   r[i] = variate_generator();
  // }
}

template
void caffe_rng_gaussian<float>(const int n, const float mu,
                               const float sigma, float* r);

template
void caffe_rng_gaussian<double>(const int n, const double mu,
                                const double sigma, double* r);

template
void caffe_rng_gaussian<_Float16>(const int n, const _Float16 mu,
                                const _Float16 sigma, _Float16* r);

// template <>
// void caffe_rng_gaussian<_Float16>(const int n, const _Float16 mu,
//                                 const _Float16 sigma, _Float16* r) {
//   std::vector<float> res(n, .0);
//   caffe_rng_gaussian<float>(n, float(mu), float(sigma), res.data());
//   for(int i = 0; i < n; ++i) {
//     r[i] = _Float16(res[i]);
//     if(i < 20) {
//       LOG(INFO) << r[i] << " " << res[i] << std::endl;
//     }
//   }
//   LOG(INFO) << "Call caffe_rng_gaussian too many time hurt performance!";
// }

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(float(p), 0);
  CHECK_LE(float(p), 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = variate_generator();
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, int* r);

template
void caffe_rng_bernoulli<_Float16>(const int n, const _Float16 p, int* r);

template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype p, unsigned int* r) {
  CHECK_GE(n, 0);
  CHECK(r);
  CHECK_GE(float(p), 0);
  CHECK_LE(float(p), 1);
  boost::bernoulli_distribution<Dtype> random_distribution(p);
  boost::variate_generator<caffe::rng_t*, boost::bernoulli_distribution<Dtype> >
      variate_generator(caffe_rng(), random_distribution);
  for (int i = 0; i < n; ++i) {
    r[i] = static_cast<unsigned int>(variate_generator());
  }
}

template
void caffe_rng_bernoulli<double>(const int n, const double p, unsigned int* r);

template
void caffe_rng_bernoulli<float>(const int n, const float p, unsigned int* r);

template
void caffe_rng_bernoulli<_Float16>(const int n, const _Float16 p, unsigned int* r);

template <>
float caffe_cpu_strided_dot<float>(const int n, const float* x, const int incx,
    const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template <>
double caffe_cpu_strided_dot<double>(const int n, const double* x,
    const int incx, const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

template <>
_Float16 caffe_cpu_strided_dot<_Float16>(const int n, const _Float16* x,
    const int incx, const _Float16* y, const int incy) {
  return cblas_hdot(n, x, incx, y, incy);
}

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y) {
  return caffe_cpu_strided_dot(n, x, 1, y, 1);
}

template
float caffe_cpu_dot<float>(const int n, const float* x, const float* y);

template
double caffe_cpu_dot<double>(const int n, const double* x, const double* y);

template
_Float16 caffe_cpu_dot<_Float16>(const int n, const _Float16* x, const _Float16* y);

template <>
float caffe_cpu_asum<float>(const int n, const float* x) {
  return cblas_sasum(n, x, 1);
}

template <>
double caffe_cpu_asum<double>(const int n, const double* x) {
  return cblas_dasum(n, x, 1);
}

template <>
_Float16 caffe_cpu_asum<_Float16>(const int n, const _Float16* x) {
  std::cout << "FOR Debug Only!" << std::endl;
  std::vector<float> x32(n, .0);
  for(int i = 0; i < n; ++i) {
    x32[i] = float(x[i]);
  }
  LOG(INFO) << "In caffe_cpu_asum " << cblas_sasum(n, x32.data(), 1);
  return _Float16(cblas_sasum(n, x32.data(), 1));
}

template <>
void caffe_cpu_scale<float>(const int n, const float alpha, const float *x,
                            float* y) {
  cblas_scopy(n, x, 1, y, 1);
  cblas_sscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<double>(const int n, const double alpha, const double *x,
                             double* y) {
  cblas_dcopy(n, x, 1, y, 1);
  cblas_dscal(n, alpha, y, 1);
}

template <>
void caffe_cpu_scale<_Float16>(const int n, const _Float16 alpha, const _Float16 *x,
                             _Float16* y) {
  std::cout << "FOR Debug Only!" << std::endl;

  // cblas_hscale(n, alpha, x, y);
  std::vector<float> x32(n);
  std::vector<float> y32(n);

  for(int i = 0; i < n; ++i) {
    x32[i] = float(x[i]);
    y32[i] = float(y[i]);
  }

  caffe_cpu_scale<float>(n, float(alpha), x32.data(), y32.data());
  for(int i = 0; i < n; ++i) {
    y[i] = _Float16(y32[i]);
  }
}

}  // namespace caffe
