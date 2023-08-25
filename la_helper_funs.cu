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
__global__ void matrixDiv(float* A, float B, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        A[i] /= B;
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
extern "C"
__global__ void mask(const float* __restrict__ A, float val, float newVal, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
       if (A[i] == val)
       {
           C[i] = newVal;
       }
    }
}
extern "C"
__global__ void dropout(float* A, float random, float chanceDrop, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    float drop = 1.0f / (1.0f - chanceDrop);
    if (i < numElements)
    {
       if (random > chanceDrop)
       {
           A[i] = A[i] * drop;
       }
    }
}
extern "C"
__global__ void derSoftmax(const float* __restrict__ output, const float* __restrict__ error,int column, float* C, int numElements)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       int index, indexI, indexJ;
       float value;
       index = k * column;
       indexI = index;
       for (int i = 0; i < column; i++, indexI++)
       {
           C[indexI] = 0;
           indexJ = index;
           for (int j = 0; j < column; j++, indexJ++) 
           {
               if (i != j) 
               {
                   value = output[indexI] * -output[indexJ];
               } 
               else 
               {
                   value = output[indexI] * (1 - output[indexI]);
               }
               C[indexI] += error[indexJ] * value;
           }
       }
   }
}
extern "C"
__global__ void MatAdd(float* A, const float* __restrict__ B, int numElements)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       A[k] += B[k];
    }
}
extern "C"
__global__ void addCopy(float* A, float* C, int A_col, int C_col, int start, int n) 
{
int index = threadIdx.x + (blockIdx.x * blockDim.x);
if (index >= n)
   return;
int indexOut = 0;
int indexIn = index * C_col + start * A_col;
for (int j = 0; j < A_col; j++, indexIn++, indexOut++) 
{
    C[indexIn] = A[indexOut];
}
}
extern "C"
__global__ void addBackCopy(float* A, float* C, int A_col, int C_col, int start, int n) 
{
   int index = threadIdx.x + (blockIdx.x * blockDim.x);
   if (index >= n)
       return;
   int indexOut = 0;
   int indexIn = index * A_col + start * C_col;
   for (int j = 0; j < C_col; j++, indexIn++, indexOut++) 
   {
       C[indexIn] = A[indexOut];
   }
}
extern "C"
__global__ void Softmax(const float* __restrict__ A, float* auxE, int sample_dim, float* N, int numElements)
{
float C_value = 0;
int thread_id_x = blockDim.x * blockIdx.x + threadIdx.x;
float maxCoef = A[thread_id_x*sample_dim];
float actualCoef = 0;
double E = 2.718281828459045;
if (thread_id_x < numElements)
{
#pragma omp parallel for
for (int cA = 1; cA < sample_dim; cA++)
if (A[thread_id_x * sample_dim + cA] > maxCoef)
maxCoef = A[thread_id_x * sample_dim+cA];
#pragma omp parallel for
for (int cA = 0; cA < sample_dim; cA++)
{
actualCoef = (float) pow(E, (double)(A[thread_id_x * sample_dim + cA] - maxCoef));
auxE[thread_id_x * sample_dim + cA] = actualCoef;
C_value += actualCoef;
}
#pragma omp parallel for
C_value += 0.00000001f;
for (int cA = 0; cA < sample_dim; cA++)
{
N[thread_id_x * sample_dim + cA] = auxE[thread_id_x * sample_dim + cA] / C_value;
}
}
}

