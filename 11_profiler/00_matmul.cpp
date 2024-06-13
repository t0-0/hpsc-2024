#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <vector>
#include "timers.h"
#include <immintrin.h>
using namespace std;
typedef vector<vector<float>> matrix;

void matmult(matrix &A, matrix &B, matrix &C, int N) {
  const int m = N, n = N, k = N;
  const int kc = 2048;
  const int nc = 64;
  const int mc = 128;
  const int nr = 64;
  const int mr = 32;
#pragma omp parallel for
  for (int jc=0; jc<n; jc+=nc) {
    for (int pc=0; pc<k; pc+=kc) {
      float __attribute__((aligned(64))) Bc[kc*nc];
      for (int p=0; p<kc; p++) {
        for (int j=0; j<nc; j++) {
          Bc[p*nc+j] = B[p+pc][j+jc];
        }
      }
      for (int ic=0; ic<m; ic+=mc) {
	float __attribute__((aligned(64))) Ac[mc*kc],Cc[mc*nc];
        for (int i=0; i<mc; i++) {
          for (int p=0; p<kc; p++) {
            Ac[i*kc+p] = A[i+ic][p+pc];
          }
          for (int j=0; j<nc; j++) {
            Cc[i*nc+j] = 0;
          }
        }
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            for (int kr=0; kr<kc; kr++) {
              for (int i=ir; i<ir+mr; i++) {
		__m512 Avec = _mm512_set1_ps(Ac[i*kc+kr]);
                for (int j=jr; j<jr+nr; j+=16) {
                  __m512 Bvec = _mm512_load_ps(Bc+kr*nc+j);
                  __m512 Cvec = _mm512_load_ps(Cc+i*nc+j);
                  Cvec = _mm512_fmadd_ps(Avec, Bvec, Cvec);
                  _mm512_store_ps(Cc+i*nc+j, Cvec);
                }
              }
            }
          }
        }
        for (int i=0; i<mc; i++) {
          for (int j=0; j<nc; j++) {
            C[i+ic][j+jc] += Cc[i*nc+j];
          }
        }
      }
    }
  }
}

int main(int argc, char **argv) {
  const int N = 8192;
  matrix A(N,vector<float>(N));
  matrix B(N,vector<float>(N));
  matrix C(N,vector<float>(N));
  matmult(A,B,C,N);
  for (int i=0; i<N; i++) {
    for (int j=0; j<N; j++) {
      A[i][j] = drand48();
      B[i][j] = drand48();
      C[i][j] = 0;
    }
  }
  startTimer();
  matmult(A,B,C,N);
  stopTimer();
  double time = getTime();
  printf("N=%d: %lf s (%lf GFlops)\n",N,time,2.*N*N*N/time/1e9);
}
