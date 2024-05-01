#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void myInit(int *bucket, int num) {
  bucket[threadIdx.x] = num;
}

__global__ void myAdd(int *key, int *bucket) {
  atomicAdd(&bucket[key[threadIdx.x]], 1);
}

__global__ void myCopy(int *bucket, int *offset) {
  offset[threadIdx.x+1] = bucket[threadIdx.x];
}

__global__ void myScan(int *offset, int *offset_, int range) {
  int i = threadIdx.x;
  for(int j=1; j<range; j<<=1) {
    offset_[i] = offset[i];
    __syncthreads();
    if(i>=j) offset[i]+=offset_[i-j];
    __syncthreads();
  }
}

__global__ void mySort(int *key, int *bucket, int *offset) {
  int index = threadIdx.x;
  int i = blockIdx.x;
  if (index<bucket[i]) {
    key[index+offset[i]] = i;
  }
}

int main() {
  int n = 50;
  int range = 5;
  int *key, *bucket, *offset, *offset_;
  cudaMallocManaged(&key, n*sizeof(int));
  cudaMallocManaged(&bucket, range*sizeof(int));
  cudaMallocManaged(&offset, range*sizeof(int));
  cudaMallocManaged(&offset_, range*sizeof(int));
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  offset[0]=0;

  myInit<<<1,range>>>(bucket,0);
  cudaDeviceSynchronize();
  myAdd<<<1,n>>>(key,bucket);
  cudaDeviceSynchronize();
  myCopy<<<1,range-1>>>(bucket,offset);
  cudaDeviceSynchronize();
  myScan<<<1,range>>>(offset,offset_,range);
  cudaDeviceSynchronize();
  mySort<<<range,n>>>(key,bucket,offset);
  cudaDeviceSynchronize();

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
