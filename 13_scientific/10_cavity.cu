#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <vector>

using namespace std;

__global__ void calcB(float *b, int nx, int ny, float rho, float dt, float dx, float dy, float *u, float *v)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index % nx;
  int j = index / nx;
  if (i == 0 || i == nx - 1 || j == 0 || j >= ny - 1)
    return;
  b[index] = rho * (1 / dt * ((u[index + 1] - u[index - 1]) / (2 * dx) + (v[index + nx] - v[index - nx]) / (2 * dy)) - ((u[index + 1] - u[index - 1]) / (2 * dx)) * ((u[index + 1] - u[index - 1]) / (2 * dx)) - 2 * ((u[index + nx] - u[index - nx]) / (2 * dy) * (v[index + 1] - v[index - 1]) / (2 * dx)) - ((v[index + nx] - v[index - nx]) / (2 * dy)) * ((v[index + nx] - v[index - nx]) / (2 * dy)));
}
__global__ void calcInnerP(float *p, float *pn, float *b, float dx, float dy, int nx, int ny)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index % nx;
  int j = index / nx;
  if (i == 0 || i == nx - 1 || j == 0 || j >= ny - 1)
    return;
  p[index] = (dy * dy * (pn[index + 1] + pn[index - 1]) + dx * dx * (pn[index + nx] + pn[index - nx]) - b[index] * dx * dx * dy * dy) / (2 * (dx * dx + dy * dy));
}
__global__ void calcOuterP(float *p, int stride, int offset, int diff)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  p[index * stride + offset] = p[index * stride + offset + diff];
}
__global__ void calcInnerU(float *u, float *un, float *p, float dt, float dx, float dy, float rho, int nx, int ny, float nu)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index % nx;
  int j = index / nx;
  if (i == 0 || i == nx - 1 || j == 0 || j >= ny - 1)
    return;
  u[index] = un[index] - un[index] * dt / dx * (un[index] - un[index - 1]) - un[index] * dt / dy * (un[index] - un[index - nx]) - dt / (2 * rho * dx) * (p[index + 1] - p[index - 1]) + nu * dt / (dx * dx) * (un[index + 1] - 2 * un[index] + un[index - 1]) + nu * dt / (dy * dy) * (un[index + nx] - 2 * un[index] + un[index - nx]);
}
__global__ void calcInnerV(float *v, float *vn, float *p, float dt, float dx, float dy, float rho, int nx, int ny, float nu)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int i = index % nx;
  int j = index / nx;
  if (i == 0 || i == nx - 1 || j == 0 || j >= ny - 1)
    return;
  v[index] = vn[index] - vn[index] * dt / dx * (vn[index] - vn[index - 1]) - vn[index] * dt / dy * (vn[index] - vn[index - nx]) - dt / (2 * rho * dx) * (p[index + nx] - p[index - nx]) + nu * dt / (dx * dx) * (vn[index + 1] - 2 * vn[index] + vn[index - 1]) + nu * dt / (dy * dy) * (vn[index + nx] - 2 * vn[index] + vn[index - nx]);
}

__global__ void setOuter(float *l, int stride, int offset, int value)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  l[index * stride + offset] = value;
}

int main()
{
  int nx = 41;
  int ny = 41;
  int nt = 500;
  int nit = 50;
  double dx = 2. / (nx - 1);
  double dy = 2. / (ny - 1);
  double dt = .01;
  double rho = 1.;
  double nu = .02;

  float *u, *v, *p, *b, *un, *vn, *pn;
  cudaMallocManaged(&u, ny * nx * sizeof(float));
  cudaMallocManaged(&v, ny * nx * sizeof(float));
  cudaMallocManaged(&p, ny * nx * sizeof(float));
  cudaMallocManaged(&b, ny * nx * sizeof(float));
  cudaMallocManaged(&un, ny * nx * sizeof(float));
  cudaMallocManaged(&vn, ny * nx * sizeof(float));
  cudaMallocManaged(&pn, ny * nx * sizeof(float));
  for (int i = 0; i < ny * nx; i++)
  {
    u[i] = 0;
    v[i] = 0;
    p[i] = 0;
    b[i] = 0;
  }
  ofstream ufile("u.dat");
  ofstream vfile("v.dat");
  ofstream pfile("p.dat");
  for (int n = 0; n < nt; n++)
  {
    calcB<<<(ny * nx + 1024) / 1024, 1024>>>(b, nx, ny, rho, dt, dx, dy, u, v);
    cudaDeviceSynchronize();
    for (int it = 0; it < nit; it++)
    {
      for (int i = 0; i < ny * nx; i++)
        pn[i] = p[i];
      calcInnerP<<<(ny * nx + 1024) / 1024, 1024>>>(p, pn, b, dx, dy, nx, ny);
      cudaDeviceSynchronize();
      calcOuterP<<<1, ny>>>(p, nx, nx - 1, -1);
      cudaDeviceSynchronize();
      calcOuterP<<<1, nx>>>(p, 1, 0, nx);
      cudaDeviceSynchronize();
      calcOuterP<<<1, ny>>>(p, nx, 0, 1);
      cudaDeviceSynchronize();
      setOuter<<<1, nx>>>(p, 1, ny * nx - nx, 0);
      cudaDeviceSynchronize();
    }
    for (int i = 0; i < ny * nx; i++)
    {
      un[i] = u[i];
      vn[i] = v[i];
    }
    calcInnerU<<<(ny * nx + 1024) / 1024, 1024>>>(u, un, p, dt, dx, dy, rho, nx, ny, nu);
    calcInnerV<<<(ny * nx + 1024) / 1024, 1024>>>(v, vn, p, dt, dx, dy, rho, nx, ny, nu);
    cudaDeviceSynchronize();
    setOuter<<<1, nx>>>(u, 1, 0, 0);
    cudaDeviceSynchronize();
    setOuter<<<1, ny>>>(u, nx, 0, 0);
    setOuter<<<1, ny>>>(u, nx, nx - 1, 0);
    cudaDeviceSynchronize();
    setOuter<<<1, nx>>>(u, 1, nx * ny - nx, 1);
    setOuter<<<1, nx>>>(v, 1, 0, 0);
    setOuter<<<1, nx>>>(v, 1, nx * ny - nx, 0);
    cudaDeviceSynchronize();
    setOuter<<<1, ny>>>(v, nx, 0, 0);
    setOuter<<<1, ny>>>(v, nx, nx - 1, 0);
    cudaDeviceSynchronize();
    if (n % 10 == 0)
    {
      for (int i = 0; i < ny * nx; i++)
        ufile << u[i] << " ";
      ufile << "\n";
      for (int i = 0; i < ny * nx; i++)
        vfile << v[i] << " ";
      vfile << "\n";
      for (int i = 0; i < ny * nx; i++)
        pfile << p[i] << " ";
      pfile << "\n";
    }
  }
  ufile.close();
  vfile.close();
  pfile.close();
}
