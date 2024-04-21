#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range,0); 
  for (int i=0; i<n; i++)
    bucket[key[i]]++;
  std::vector<int> offset(range,0);
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
#pragma omp parallel for
  for (int i=0; i<n; i++)
    key[i]=-1;
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    std::vector<int> b(bucket[i],0);
#pragma omp parallel for
    for(int j=0; j<b.size(); j++)
      b[j] = offset[i]+j;
#pragma omp parallel for
    for (int j=0; j<b.size(); j++) {
#pragma omp atomic write
      key[b[j]] = std::max(i,key[b[j]]);
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
