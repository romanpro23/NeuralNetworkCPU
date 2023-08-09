extern "C"
__global__ void vectorScalarSet(float* A, float alpha, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        A[i] = alpha;
    }
}
extern "C"
__global__ void vectorScalarAdd(const float* __restrict__ A, float* B, float alpha, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = A[i] + alpha;
    }
}
extern "C"
__global__ void vectorLog(const float* __restrict__ A, float* B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = log(A[i]);
    }
}
extern "C"
__global__ void vectorExp(const float* __restrict__ A, float* B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = exp(A[i]);
    }
}
extern "C"
__global__ void vectorSign(const float* __restrict__ A, float* B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = (A[i] > 0.0 ? 1.0 : -1.0);
    }
}
extern "C"
__global__ void vectorAbs(const float* __restrict__ A, float* B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = abs(A[i]);
    }
}
extern "C"
__global__ void vectorDiv(const float* __restrict__ A, const float* __restrict__ B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] / B[i];
    }
}
extern "C"
__global__ void vectorMul(const float* __restrict__ A, const float* __restrict__ B, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = A[i] * B[i];
    }
}
extern "C"
__global__ void vectorMax(const float* __restrict__ A, float* B, float val, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = max(A[i], val);
    }
}
extern "C"
__global__ void vectorMin(const float* __restrict__ A, float* B, float val, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = min(A[i], val);
    }
}
extern "C"
__global__ void vectorPow(const float* __restrict__ A, float* B, float val, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = pow((double) A[i], (double) val);
    }
}
extern "C"
__global__ void vectorSqr(const float* __restrict__ A, float* B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float val;
    if (i < numElements)
    {
        val = A[i];
        B[i] = val*val;
    }
}
extern "C"
__global__ void vectorSqrt(const float* __restrict__ A, float* B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        B[i] = sqrt(A[i]);
    }
}
