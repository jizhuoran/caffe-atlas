#ifndef ARMBLAS_F16_HPP
#define ARMBLAS_F16_HPP

/*
 * ===========================================================================
 * Prototypes for level 1 BLAS routines
 * ===========================================================================
 */

_Float16  cblas_hdot(const int N, const _Float16  *X, const int incX,
    const _Float16  *Y, const int incY);

void cblas_hscal(const int N, const _Float16 alpha, _Float16 *X, const int incX);

void cblas_hscale(const int N, const _Float16 alpha, const _Float16 *X, _Float16 *Y);

void cblas_haxpy(const int N, const _Float16 alpha, const _Float16 *X,
    const int incX, _Float16 *Y, const int incY); 
void cblas_haxpby(const int N, const _Float16 alpha, const _Float16 *X,
    const int incX, const _Float16 beta, _Float16 *Y, const int incY);


// void cblas_hswap(const int N, float *X, const int incX,
//                  float *Y, const int incY);
// void cblas_hcopy(const int N, const float *X, const int incX,
//                  float *Y, const int incY);


// void catlas_sset
//    (const int N, const float alpha, float *X, const int incX);

// void cblas_dswap(const int N, double *X, const int incX,
//                  double *Y, const int incY);
// void cblas_dcopy(const int N, const double *X, const int incX,
//                  double *Y, const int incY);
// void cblas_daxpy(const int N, const double alpha, const double *X,
//                  const int incX, double *Y, const int incY);
// void catlas_daxpby(const int N, const double alpha, const double *X,
//                   const int incX, const double beta, double *Y, const int incY);
// void catlas_dset
//    (const int N, const double alpha, double *X, const int incX);

// void cblas_cswap(const int N, void *X, const int incX,
//                  void *Y, const int incY);
// void cblas_ccopy(const int N, const void *X, const int incX,
//                  void *Y, const int incY);
// void cblas_caxpy(const int N, const void *alpha, const void *X,
//                  const int incX, void *Y, const int incY);
// void catlas_caxpby(const int N, const void *alpha, const void *X,
//                   const int incX, const void *beta, void *Y, const int incY);
// void catlas_cset
//    (const int N, const void *alpha, void *X, const int incX);

// void cblas_zswap(const int N, void *X, const int incX,
//                  void *Y, const int incY);
// void cblas_zcopy(const int N, const void *X, const int incX,
//                  void *Y, const int incY);
// void cblas_zaxpy(const int N, const void *alpha, const void *X,
//                  const int incX, void *Y, const int incY);
// void catlas_zaxpby(const int N, const void *alpha, const void *X,
//                   const int incX, const void *beta, void *Y, const int incY);
// void catlas_zset
//    (const int N, const void *alpha, void *X, const int incX);



#endif // ARMBLAS_F16_HPP