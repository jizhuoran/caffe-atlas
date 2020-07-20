#include "caffe/util/armblas_fp16.hpp"

namespace {

void cblas_hzero(const int N, _Float16 *X, const int incX)
/*
 * X <- 0
 */
{
   int i, n;
   if (incX == 1)
   {
      n = N;
      i = n >> 5;
      if (i)
      {
         n -= (i << 5);
         do
         {
            *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = X[8] = X[9] =
            X[10] = X[11] = X[12] = X[13] = X[14] = X[15] = X[16] = X[17] =
            X[18] = X[19] = X[20] = X[21] = X[22] = X[23] = X[24] = X[25] =
            X[26] = X[27] = X[28] = X[29] = X[30] = X[31] = _Float16(.0);
            X += 32;
         }
         while(--i);
      }
      if (n >> 4) /* >= 16 */
      {
         *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = X[8] = X[9] =
         X[10] = X[11] = X[12] = X[13] = X[14] = X[15] = _Float16(.0);
         X += 16;
         n -= 16;
      }
      if (n >> 3) /* >= 8 */
      {
         *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = X[7] = _Float16(.0);
         X += 8;
         n -= 8;
      }
      switch(n)
      {
         case 1:
            *X = _Float16(.0);
            break;
         case 2:
            *X = X[1] = _Float16(.0);
            break;
         case 3:
            *X = X[1] = X[2] = _Float16(.0);
            break;
         case 4:
            *X = X[1] = X[2] = X[3] = _Float16(.0);
            break;
         case 5:
            *X = X[1] = X[2] = X[3] = X[4] = _Float16(.0);
            break;
         case 6:
            *X = X[1] = X[2] = X[3] = X[4] = X[5] = _Float16(.0);
            break;
         case 7:
            *X = X[1] = X[2] = X[3] = X[4] = X[5] = X[6] = _Float16(.0);
            break;
         default:;
      }
   }
   else
   {
        for (i=N; i; i--, X += incX) *X = _Float16(.0);
   }
}
}






_Float16  cblas_hdot(const int N, const _Float16  *X, const int incX,
    const _Float16  *Y, const int incY) {
    _Float16 dot = _Float16{.0};
    if (incX == 1 && incY == 1)
        for (int i = 0; i < N; i++) {
            dot += X[i] * Y[i];
        }
    else {
        for (int i = 0; i < N; i++, X += incX, Y += incY) {
            dot += *X * *Y;
        }
    }
    return dot;
}


void cblas_hscal(const int N, const _Float16 alpha, _Float16 *X, const int incX) {
    if (alpha == _Float16(.0)) {
        cblas_hzero(N, X, incX);
    } else if(alpha == _Float16(1.)) {
        return;
    }else {
        if (incX == 1) for (int i=0; i != N; i++) X[i] *= alpha;
        else for (int i=0; i < N; i++, X += incX) *X *= alpha;
    }
}


void cblas_haxpy(const int N, const _Float16 alpha, const _Float16 *X,
    const int incX, _Float16 *Y, const int incY) {
   if (alpha != _Float16(.0))
   {
      if (alpha == _Float16(1.))
      {
         if (incX == 1 && incY == 1)
            for (int i=0; i != N; i++) Y[i] += X[i];
         else for (int i=0; i < N; i++, X += incX, Y += incY) *Y += *X;
      } else if (incX == 1 && incY == 1)
        for (int i=0; i != N; i++) Y[i] += alpha * X[i];
      else for (int i=0; i < N; i++, X += incX, Y += incY) *Y += alpha * *X;
   }
}

void cblas_haxpby(const int N, const _Float16 alpha, const _Float16 *X,
    const int incX, const _Float16 beta, _Float16 *Y, const int incY) {

    if (alpha == _Float16(.0))
    {
        if (beta != _Float16(.0))
            cblas_hscal(N, beta, Y, incY);
        else
            cblas_hzero (N, Y, incY);  
    }
    else if (beta == _Float16(.0)) {
        if(alpha == _Float16(1.)) {
            if (incX == 1 && incY == 1)
                for (int i=0; i != N; i++) Y[i] = X[i];
            else for (int i=0; i < N; i++, X += incX, Y += incY) *Y = *X;
        } else { //alpha != 0 != 1
            if (incX == 1 && incY == 1)
                for (int i=0; i != N; i++) Y[i] = alpha * X[i];
            else for (int i=0; i < N; i++, X += incX, Y += incY) *Y = alpha * *X;
        }
    }
    else if (beta == _Float16(1.)) cblas_haxpy(N, alpha, X, incX, Y, incY);
    else if (alpha == _Float16(1.)) {
        if (incX == 1 && incY == 1)
            for (int i=0; i != N; i++) Y[i] = X[i] + beta * Y[i];
        else for (int i=0; i < N; i++, X += incX, Y += incY) *Y = *X + (beta * *Y);
    } else if (incX == 1 && incY == 1) {
        for (int i=0; i != N; i++) Y[i] = alpha * X[i] + beta * Y[i];
    }
    else for (int i=0; i < N; i++, X += incX, Y += incY) *Y = (alpha * *X) + (beta * *Y);
}






