#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }

  __m512 xvec = _mm512_load_ps(x);
  __m512 yvec = _mm512_load_ps(y);
  __m512 mvec = _mm512_load_ps(m);
  float z2s[N] = {0, 1, 2, 3, 4, 5, 6, 7};
  __m512 z2svec = _mm512_load_ps(z2s);

  for(int i=0; i<N; i++) {
    __m512 is = _mm512_set1_ps(i);
    __mmask16 mask = _mm512_cmp_ps_mask(z2svec, is, _MM_CMPINT_NE);

    __m512 rxvec = _mm512_sub_ps(_mm512_set1_ps(x[i]), xvec);

    __m512 ryvec = _mm512_sub_ps(_mm512_set1_ps(y[i]), yvec);

    __m512 rrvec = _mm512_rsqrt14_ps(_mm512_add_ps(_mm512_mul_ps(rxvec, rxvec), _mm512_mul_ps(ryvec, ryvec)));

    float dfx = _mm512_reduce_add_ps(_mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(rxvec, _mm512_mul_ps(mvec, _mm512_mul_ps(rrvec, _mm512_mul_ps(rrvec, rrvec))))));

    float dfy = _mm512_reduce_add_ps(_mm512_mask_blend_ps(mask, _mm512_setzero_ps(), _mm512_mul_ps(ryvec, _mm512_mul_ps(mvec, _mm512_mul_ps(rrvec, _mm512_mul_ps(rrvec, rrvec))))));

    fx[i] -= dfx;
    fy[i] -= dfy;
    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
