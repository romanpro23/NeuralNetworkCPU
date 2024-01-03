#include <cuda_bf16.h>
#include <float.h>
#define TYPE __nv_bfloat16
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64
__device__ const int SharedMemorySize = 64 * 1024 / 2;
__device__ const int BLOCK_DIM = 32;
__device__ const int TILE_WIDTH = 32;
__device__ __constant__ TYPE sh[17];
__inline__ __device__ float InfinityCheck(float v)
{
    int r = __isinf(v);
    if (r == 1) {
        v = FLT_MAX;
    }
    else if (r == -1) {
        v = -FLT_MAX;
    }
    return v;
}
__inline__ __device__ float InfinityCheck_TYPE(TYPE v)
{
    int r = __isinf(__bfloat162float(v));
    if (r == 1) {
        v = __float2bfloat16(FLT_MAX);
    }
    else if (r == -1) {
        v = __float2bfloat16(-FLT_MAX);
    }
    return v;
}
extern "C"
__global__ void fill(TYPE* A, TYPE alpha, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = alpha;
    }
}
extern "C"
__global__ void fill_float(float* A, float alpha, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = alpha;
    }
}
extern "C"
__global__ void float2TYPEVector(float* v, TYPE* data, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        data[i] = __float2bfloat16(v[i]);
    }
}
extern "C"
__global__ void TYPE2FloatVector(TYPE* v, float* data, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        data[i] = __bfloat162float(v[i]);
    }
}
extern "C"
__global__ void gelu(const float* __restrict__ A, float* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float a = A[i];
        float t = tanhf(0.7978846f * a + 0.0356774f * (a * a * a));
        C[i] = 0.5f * a * (1.0f + t);
    }
}
extern "C"
__global__ void gelu_TYPE(const TYPE* __restrict__ A, TYPE* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float a = __bfloat162float(A[i]);
        float t = tanhf(0.7978846f * a + 0.0356774f * (a * a * a));
        C[i] = __float2bfloat16(0.5f * a * (1.0f + t));
    }
}
extern "C"
__global__ void matAdd(float* A, const float* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = A[k] + B[k];
    }
}
extern "C"
__global__ void matAdd_TYPE(TYPE* A, const TYPE* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = A[k] + B[k];
    }
}
extern "C"
__global__ void matAdd_(float* A, const float* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = A[k] + B[k];
    }
}
extern "C"
__global__ void matAdd_TYPE_(TYPE* A, const TYPE* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = A[k] + B[k];
    }
}
extern "C"
__global__ void imageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)
{
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;
    const int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (h < rows && w < columns && z < sizeKernel)
    {
        for (int k = 0; k < sizeKernel; k++) {
            int indexInput = (h + z) * depth * columns + (w + k) * depth;
            for (int c = 0; c < depth; c++, indexInput++) {
                int index = h * columns + w * sizeKernel + z * sizeKernel + k * depth + c;
                C[index] = A[indexInput];
            }
        }
    }
}
extern "C"
__global__ void backImageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)
{
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;
    const int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (h < rows && w < columns && z < sizeKernel)
    {
        for (int k = 0; k < sizeKernel; k++) {
            int indexInput = (h + z) * depth * columns + (w + k) * depth;
            for (int c = 0; c < depth; c++, indexInput++) {
                int index = h * columns + w * sizeKernel + z * sizeKernel + k * depth + c;
                C[indexInput] = A[index];
            }
        }
    }
}
extern "C"
__global__ void add3(const float* __restrict__ A, float* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    if (h < rows && w < columns) {
       int index = h * columns + w;
       C[index] = C[index] + A[w];
    }
}
extern "C"
__global__ void add3_TYPE(const TYPE* __restrict__ A, TYPE* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    if (h < rows && w < columns) {
       int index = h * columns + w;
       C[index] = C[index] + A[w];
    }
}
extern "C"
__global__ void dotT_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < columns) {
       float sum = sh[0];
       for (int i = 0; i < rows; i++) {
            int index = i * columns + j;
            sum = sum + A[i] * B[index];
       }
       C[j] = sum;
    }
}
extern "C"
__global__ void toHotVector(const float* __restrict__ batch, float* arr, int col, int n)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < n) {
       arr[j * col + ((int) batch[j])] = 1;
    }
}
extern "C"
__global__ void dotT_VectorAndMatrix_TYPE(const TYPE* __restrict__ A, const TYPE* __restrict__ B, TYPE* C, int rows, int columns)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < columns) {
       TYPE sum = sh[0];
       for (int i = 0; i < rows; i++) {
            int index = i * columns + j;
            sum = sum + A[i] * B[index];
       }
       C[j] = sum;
    }
}
extern "C"
__global__ void derivativeWeight(const float* __restrict__ input, const float* __restrict__ error, float* derWeight, int rows, int columns)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < rows && j < columns) {
       const int index = i * columns + j;
       derWeight[index] = derWeight[index] + error[i] * input[j];
    }
}
extern "C"
__global__ void derivativeWeight_TYPE(const TYPE* __restrict__ input, const TYPE* __restrict__ error, TYPE* derWeight, int rows, int columns)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < rows && j < columns) {
       const int index = i * columns + j;
       derWeight[index] = derWeight[index] + error[i] * input[j];
    }
}
extern "C"
__global__ void addMatrix(const float* __restrict__ matrix, float* data, int width, int depth)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < depth) {
       float d = data[k];
       for (int i = 0; i < width; i++) { 
           int	index = i * depth + k;
           d = d + matrix[index];
       }
       data[k] = d;
    }
  }
extern "C"
__global__ void addMatrix_TYPE(const TYPE* __restrict__ matrix, TYPE* data, int width, int depth)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < depth) {
       TYPE d = data[k];
       for (int i = 0; i < width; i++) { 
           int	index = i * depth + k;
           d = d + matrix[index];
       }
       data[k] = d;
    }
  }
__device__ void atomicMax(float* const address, const float value)
{
    if (*address >= value)
    {
        return;
    }
    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;
    do
    {
        assumed = old;
        if (__int_as_float(assumed) >= value)
        {
            break;
        }
        old = atomicCAS(addressAsI, assumed, __float_as_int(value));
    } while (assumed != old);
}
__device__ void atomicMax(TYPE* const address, const TYPE value)
{
    if (*address >= value)
    {
        return;
    }
    int* const addressAsI = (int*)address;
    int old = *addressAsI, assumed;
    do
    {
        assumed = old;
        if (__float2bfloat16(__int_as_float(assumed)) >= value)
        {
            break;
        }
        old = atomicCAS(addressAsI, assumed, __float_as_int(__bfloat162float(value)));
    } while (assumed != old);
}
extern "C"
__global__ void reduceMaxIdxOptimizedShared(const float* __restrict__ input, const int size, float* maxOut, int* maxIdxOut)
{
    __shared__ float sharedMax;
    __shared__ int sharedMaxIdx;
    if (0 == threadIdx.x)
    {
        sharedMax = 0.f;
        sharedMaxIdx = 0;
    }
    __syncthreads();
    float localMax = 0.f;
    int localMaxIdx = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        float val = input[i];
        if (localMax < val)
        {
            localMax = val;
            localMaxIdx = i;
        }
    }
    atomicMax(&sharedMax, localMax);
    __syncthreads();
    if (sharedMax == localMax)
    {
        sharedMaxIdx = localMaxIdx;
    }
    __syncthreads();
    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
        *maxIdxOut = sharedMaxIdx;
    }
}
extern "C"
__global__ void reduceMaxIdxOptimizedShared_TYPE(const TYPE* __restrict__ input, const int size, TYPE* maxOut, int* maxIdxOut)
{
    __shared__ TYPE sharedMax;
    __shared__ int sharedMaxIdx;
    if (0 == threadIdx.x)
    {
        sharedMax = sh[0];
        sharedMaxIdx = 0;
    }
    __syncthreads();
    TYPE localMax = sh[0];
    int localMaxIdx = 0;
    for (int i = threadIdx.x; i < size; i += blockDim.x)
    {
        TYPE val = input[i];
        if (localMax < val)
        {
            localMax = val;
            localMaxIdx = i;
        }
    }
    atomicMax(&sharedMax, localMax);
    __syncthreads();
    if (sharedMax == localMax)
    {
        sharedMaxIdx = localMaxIdx;
    }
    __syncthreads();
    if (0 == threadIdx.x)
    {
        *maxOut = sharedMax;
        *maxIdxOut = sharedMaxIdx;
    }
}
extern "C"
__global__ void reverse(TYPE* A, int rows, int columns, int depth)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int k = blockDim.z * blockIdx.z + threadIdx.z;
    const int index = i * blockDim.y * gridDim.y + j;
    if (index < rows * columns  && (k < depth))
    {
       const int index = rows - 1 - i;
       TYPE valf = A[i * depth * columns + j * depth + k];
       TYPE vals = A[index  * depth * columns + j * depth + k];
       A[i  * depth * columns + j * depth + k] = valf;
       A[index  * depth * columns + j * depth + k] = vals;
    }
}
extern "C"
__global__ void sharedMem_transpose(float* R, float* M, int rows, int cols){
    __shared__ float M_Shared[BLOCK_DIM][BLOCK_DIM];
    int tile_size = BLOCK_DIM;
    int idx = tile_size * blockIdx.x + threadIdx.x;
    int idy = tile_size * blockIdx.y + threadIdx.y;
    int index_in = idx * cols + idy;
    int index_out = idy * rows + idx;
    if (idx < rows && idy < cols) {
        M_Shared[threadIdx.y][threadIdx.x] = M[index_in];
    }
    __syncthreads();
    if(idx < rows && idy < cols){
        R[index_out] = M_Shared[threadIdx.y][threadIdx.x];
    }
}
extern "C"
__global__ void sharedMem_transpose_TYPE(TYPE* R, TYPE* M, int rows, int cols){
    __shared__ TYPE M_Shared[BLOCK_DIM][BLOCK_DIM];
    int tile_size = BLOCK_DIM;
    int idx = tile_size * blockIdx.x + threadIdx.x;
    int idy = tile_size * blockIdx.y + threadIdx.y;
    int index_in = idx * cols + idy;
    int index_out = idy * rows + idx;
    if (idx < rows && idy < cols) {
        M_Shared[threadIdx.y][threadIdx.x] = M[index_in];
    }
    __syncthreads();
    if(idx < rows && idy < cols){
        R[index_out] = M_Shared[threadIdx.y][threadIdx.x];
    }
}
extern "C"
__global__ void matrixDiv(TYPE* A, TYPE B, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = A[i] / B;
    }
}
extern "C"
__global__ void matrixDiv_float(float* A, float B, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] /= B;
    }
}
extern "C"
__global__ void addCopy(const float* __restrict__ matrix, float* data, int row, int col, int m_col, int start) 
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < row && y < m_col)
    {
        const int indexIn = x * col + start * m_col + y;
        const int indexOut = x * m_col + y;
        data[indexIn] = matrix[indexOut];
    }
}
extern "C"
__global__ void addCopy_TYPE(const TYPE* __restrict__ matrix, TYPE* data, int row, int col, int m_col, int start) 
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < row && y < m_col)
    {
        const int indexIn = x * col + start * m_col + y;
        const int indexOut = x * m_col + y;
        data[indexIn] = matrix[indexOut];
    }
}
extern "C"
__global__ void NormalizationLayerForward2D(float*** P, const float* __restrict__ gamma, const float* __restrict__ betta, int width, int depth, int n)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < width) {
        float* input = P[0][x];
        float mean = P[1][x][y];
        int index = y * depth;
        for (int k = 0; k < depth; k++, index++) {
           mean = mean + input[index];
        }
        mean = mean / depth;
        P[1][x][y] = mean;
        float var = P[2][x][y];
        float sub;
        index = y * depth;
        mean = P[1][x][y];
        for (int k = 0; k < depth; k++, index++) {
            sub = input[index] - mean;
            var = var + sub * sub;
        }
        var = var / depth;
        P[2][x][y] = var;
        float varSqrt = sqrtf(var + 0.001f);
        float* output = P[3][x];
        index = y * depth;
        for (int k = 0; k < depth; k++, index++) {
             output[index] = ((input[index] - mean) / varSqrt) * gamma[k] + betta[k];
        }
    }
}
extern "C"
__global__ void NormalizationLayerBackward2D(float*** P, const float* __restrict__ gamma, const float* __restrict__ betta, float* derGamma, float* derBetta, int outWidth, int outDepth, int width, int depth, int n)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < width) {
        float* errorNL = P[0][x];
        float var = P[1][x][y];
        float* input = P[2][x];
        float mean = P[3][x][y];
        float* error = P[4][x];
        float* output = P[5][x];
        float dVar_m = -0.5f * powf(var + 0.001f, -1.5f);
        int index = y * depth;
        float derVariance = 0.0f;
        for (int k = 0; k < depth; k++, index++) {
            derVariance += errorNL[index] * gamma[k] * (input[index] - mean);
        }
        derVariance *= dVar_m;
        dVar_m = 0.0f;
        float derMean = 0.0f;
        float dMean = -1.0f / sqrtf(var + 0.001f);
        index = y * depth;
        for (int k = 0; k < depth; k++, index++) {
            derMean += errorNL[index] * gamma[k];
            dVar_m += input[index] - mean;
        }
        derMean *= dMean;
        derMean += (-2.0f * derVariance * dVar_m) / depth;
        derMean /= depth;
        derVariance *= 2.0f / (depth);
        float _dVar = 1.0f / sqrtf(var + 0.001f);
        index = y * depth;
        for (int k = 0; k < depth; k++, index++) {
            error[index] = errorNL[index] * gamma[k] * _dVar + derVariance * (input[index] - mean) + derMean;
        }
        index = y * depth;
        for (int k = 0; k < depth; k++, index++) {
            atomicAdd(&derBetta[k], errorNL[index]);
            atomicAdd(&derGamma[k], errorNL[index] * ((output[index] - betta[k]) / gamma[k]));
        }
    }
}
extern "C"
__global__ void dropout(TYPE* A, TYPE* random, TYPE chanceDrop, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       TYPE drop = sh[4] / (sh[4] - chanceDrop);
       if (random[idx] > chanceDrop)
       {
           A[idx] = A[idx] * drop;
       }
    }
}
extern "C"
__global__ void sub_gpu(const float* __restrict__ first, const float* __restrict__ second, float* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = first[idx] - second[idx];
    }
}
extern "C"
__global__ void sub_gpu_TYPE(const TYPE* __restrict__ first, const TYPE* __restrict__ second, TYPE* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = first[idx] - second[idx];
    }
}
extern "C"
__global__ void sub_bfloatAndFloat(const TYPE* __restrict__ first, const float* __restrict__ second, float* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = __bfloat162float(first[idx]) - second[idx];
    }
}
extern "C"
__global__ void sub_floatAndbFloat(const float* __restrict__ first, const TYPE* __restrict__ second, float* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = first[idx] - __bfloat162float(second[idx]);
    }
}
extern "C"
__global__ void mul(TYPE* result, TYPE val, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = result[idx] * val;
    }
}
extern "C"
__global__ void mul_float(float* result, float val, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] *= val;
    }
}
extern "C"
__global__ void clip(float* data, float min, float max, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        float a = data[idx];
        if (a > max) {
            data[idx] = max;
        } else if (a < min) {
            data[idx] = min;
        }
    }
}
extern "C"
__global__ void clip_TYPE(TYPE* data, TYPE min, TYPE max, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        TYPE a = data[idx];
        if (a > max) {
            data[idx] = max;
        } else if (a < min) {
            data[idx] = min;
        }
    }
}
extern "C"
__global__ void pow2(float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] *= data[idx];
    }
}
extern "C"
__global__ void pow2_TYPE(TYPE* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = data[idx] * data[idx];
    }
}
extern "C"
__global__ void subAbs(float* first, float* second, float* result, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       result[i] = fabsf(first[i] - second[i]);
    }
}
extern "C"
__global__ void sum(float* data, float* result, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       atomicAdd(&result[0], data[i]);
    }
}
extern "C"
__global__ void derAbs(float* first, float* second, float* result, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       float diff = first[i] - second[i];
       result[i] = diff / fabsf(diff + 0.00000001f);
    }
}
extern "C"
__global__ void fisnan(const TYPE* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (isnan(__bfloat162float(data[idx]))) {
               result[0] = idx;
           }
       }
    }
}
extern "C"
__global__ void fisnan_float(const float* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (__isnan(data[idx])) {
               result[0] = idx;
           }
       }
    }
}
extern "C"
__global__ void hisinf(const TYPE* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (isinf(__bfloat162float(data[idx]))) {
               result[0] = idx;
           }
           if (isinf(__bfloat162float(data[idx])) == -1) {
               result[0] = idx;
           }
       }
    }
}
extern "C"
__global__ void hisinf_float(const float* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (__isinf(data[idx])) {
               result[0] = idx;
           }
           if (__isinf(data[idx]) == -1) {
               result[0] = idx;
           }
       }
    }
}
extern "C"
__global__ void momentum(TYPE* data, const TYPE* __restrict__ array, TYPE decay, TYPE rt, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + array[idx] * rt;
    }
}
extern "C"
__global__ void momentum_float(float* data, const float* __restrict__ array, float decay, float rt, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + array[idx] * rt;
    }
}
extern "C"
__global__ void momentumPow2(TYPE* data, const TYPE* __restrict__ vector, TYPE decay, TYPE dr, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + dr * vector[idx] * vector[idx];
    }
}
extern "C"
__global__ void momentumPow2_float(float* data, const float* __restrict__ vector, float decay, float dr, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + dr * vector[idx] * vector[idx];
    }
}
extern "C"
__global__ void subDivSqrtNorm_TYPE(const TYPE* __restrict__ nominator, const TYPE* __restrict__ denominator, TYPE lr, TYPE normN, TYPE normD, TYPE* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       TYPE sh5 = sh[5];
       TYPE cur_lr = lr / (normN +  sh5);
       data[idx] = data[idx] - (cur_lr * (nominator[idx]) / (__float2bfloat16(sqrtf(__bfloat162float(denominator[idx] / normD))) + sh5));
    }
}
extern "C"
__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       float cur_lr = lr / (normN + 0.0000001f);
       data[idx] -= cur_lr * (nominator[idx]) / (sqrtf(denominator[idx] / normD) + 0.0000001f);
    }
}
extern "C"
__global__ void addBackCopy(const float* __restrict__ matrix, int m_column, int row, int column, int start, float* data)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < row && y < column) {
       const int indexOut = x * m_column + start * column + y;
       const int indexIn = x * column + y;
       data[indexIn] = matrix[indexOut];
    }
}
extern "C"
__global__ void addBackCopy_TYPE(const TYPE* __restrict__ matrix, int m_column, int row, int column, int start, TYPE* data)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < row && y < column) {
       const int indexOut = x * m_column + start * column + y;
       const int indexIn = x * column + y;
       data[indexIn] = matrix[indexOut];
    }
}
extern "C"
__global__ void dropoutBack(const TYPE* __restrict__ output, const TYPE* __restrict__ error, TYPE drop, TYPE* data, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (output[idx] != sh[0]) {
            data[idx] = error[idx] * drop;
       };
    }
}
extern "C"
__global__ void mask(const float* __restrict__ A, float val, float newVal, float* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       if (A[i] == val) {
           C[i] = newVal;
       }
    }
}
extern "C"
__global__ void mask_TYPE(const TYPE* __restrict__ A, TYPE val, TYPE newVal, TYPE* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       if (A[i] == val) {
           C[i] = newVal;
       }
    }
}
extern "C"
__global__ void fillUnderDiagonal(int column, float val, float* data, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        for (int j = 0; j < i + 1; j++) {
            data[i * column + j] = val;
        }
    }
}
extern "C"
__global__ void crossEntropy(float* first, float* second, float* result, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        result[i] = (float) (first[i] * log(second[i] + 0.00000001f));
    }
}
extern "C"
__global__ void derCrossEntropy(float* idealOutputs, float* outputs, float* result, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        result[i] = -idealOutputs[i] / (outputs[i] + 0.00000001f);
    }
}
extern "C"
__global__ void stride(const float* __restrict__ data, float* result, int stride, float row, float column)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < row) {
        int inputIndex = i * column;
        int outputIndex = i * stride * column;
        for (int k = 0; k < column; k++, inputIndex++, outputIndex++) {
            result[outputIndex] = data[inputIndex];
        }
    }
}
extern "C"
__global__ void fillUnderDiagonal_TYPE(int column, TYPE val, TYPE* data, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        for (int j = 0; j < i + 1; j++) {
            data[i * column + j] = val;
        }
    }
}
extern "C"
__global__ void derGelu(const float* __restrict__ input, const float* __restrict__ error, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        float x = input[idx];
        float val = tanhf(0.7978846f * x + 0.0356774f * x * x * x);
        data[idx] = error[idx] * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f * x * x));
    }
}
extern "C"
__global__ void derGelu_TYPE(const TYPE* __restrict__ input, const TYPE* __restrict__ error, TYPE* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        float x = __bfloat162float(input[idx]);
        float val = tanhf(0.7978846f * x + 0.0356774f * x * x * x);
        data[idx] = __float2bfloat16(__bfloat162float(error[idx]) * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f* x * x)));
    }
}
__device__ const int BLOCK_SIZE = 32;
extern "C"
__global__ void matvec_kernel(const float * __restrict__ dA, const float * __restrict__ dx, float * __restrict__ dy, const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ float x_shared[BLOCK_SIZE];
    float y_val = 0.0;
    #pragma unroll
    for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)
    {
        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) 
           x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        else
            x_shared[threadIdx.x] = 0.f;
        __syncthreads();
    #pragma unroll
        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
        y_val = y_val + dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
    }
        __syncthreads();
    }
    if (tid < nRows) dy[tid] = y_val;
}
extern "C"
__global__ void dot_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < rows) {
       double s = 0.0f;
       for (int j = 0; j < columns; j++) {
           s += A[j] * B[h * columns + j];
       }
       C[h] = s;
    }
}
extern "C"
__global__ void matvec_kernel_TYPE(const TYPE* __restrict__ dA, const TYPE* __restrict__ dx, TYPE* __restrict__ dy, const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ TYPE x_shared[BLOCK_SIZE];
    TYPE y_val = 0.0;
    #pragma unroll
    for (unsigned int m = 0; m < ((nCols + BLOCK_SIZE - 1)/ BLOCK_SIZE); ++m)
    {
        if ((m * BLOCK_SIZE + threadIdx.x) <  nCols) 
           x_shared[threadIdx.x] = dx[threadIdx.x + m * BLOCK_SIZE];
        else
            x_shared[threadIdx.x] = 0.f;
        __syncthreads();
    #pragma unroll
        for (unsigned int e = 0; e < BLOCK_SIZE; ++e) {
        y_val = y_val + dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
    }
        __syncthreads();
    }
    if (tid < nRows) dy[tid] = y_val;
}
extern "C"
__global__ void Softmax(const float* __restrict__ input, float* data, int column, int numElements)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       float sum = 0.0f;
       int index = k * column;
       float max = input[index];
       for (int i = 1; i < column; i++, index++) {
           float in = input[index];
           if (max < in)
               max = in;
       }
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           float d = expf(input[index] - max);
           d = InfinityCheck(d);
           data[index] = d;
           sum += d;
       }
       if (sum == 0.0f) {
           sum = sum + sh[5];
       }
       sum = InfinityCheck(sum);
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           data[index] /= sum;
       }
    }
}
extern "C"
__global__ void Softmax_TYPE(const TYPE* __restrict__ input, TYPE* data, int column, int numElements)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       TYPE sum = sh[0];
       int index = k * column;
       TYPE max = input[index];
       for (int i = 1; i < column; i++, index++) {
           TYPE in = input[index];
           if (max < in)
               max = in;
       }
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           TYPE d = exp(input[index] - max);
           d = InfinityCheck_TYPE(d);
           data[index] = d;
           sum = sum + d;
       }
       if (sum == sh[0]) {
           sum = sum + sh[5];
       }
       sum = InfinityCheck_TYPE(sum);
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           data[index] = data[index] / sum;
       }
    }
}
extern "C"
__global__ void matrixMulti(float* A_d, float* B_d, float* C_d, int m, int k, int n)
{
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    float sum = 0;
    for(int t=0; t<(n-1)/TILE_WIDTH+1; t++)
    {
        if(row<m && t*TILE_WIDTH+tx<n)
            ds_A[ty][tx] = A_d[row*n + t*TILE_WIDTH+tx];
        else
            ds_A[ty][tx] = 0.0;
        if(t*TILE_WIDTH+ty<n && col<k)
            ds_B[ty][tx] = B_d[(t*TILE_WIDTH+ty)*k + col];
        else
            ds_B[ty][tx] = 0.0;
        __syncthreads();
        for(int i=0; i<TILE_WIDTH; i++)
            sum += ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }
    if(row<m && col<k)
        C_d[col+row*k] = sum;
}
extern "C"
__global__ void matrixMulti_TYPE(TYPE* A_d, TYPE* B_d, TYPE* C_d, int m, int k, int n)
{
    __shared__ TYPE ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ TYPE ds_B[TILE_WIDTH][TILE_WIDTH];
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    TYPE sum = sh[0];
    for(int t=0; t<(n-1)/TILE_WIDTH+1; t++)
    {
        if(row<m && t*TILE_WIDTH+tx<n)
            ds_A[ty][tx] = A_d[row*n + t*TILE_WIDTH+tx];
        else
            ds_A[ty][tx] = 0.0;
        if(t*TILE_WIDTH+ty<n && col<k)
            ds_B[ty][tx] = B_d[(t*TILE_WIDTH+ty)*k + col];
        else
            ds_B[ty][tx] = 0.0;
        __syncthreads();
        for(int i=0; i<TILE_WIDTH; i++)
            sum = sum + ds_A[ty][i] * ds_B[i][tx];
        __syncthreads();
    }
    if(row<m && col<k)
        C_d[col+row*k] = sum;
}
#define Mask_width 5 
#define Mask_radius Mask_width / 2
#define TILE_WIDTH 16
#define w (TILE_WIDTH + Mask_width - 1)
#define clamp(x) (min(max((x), 0.0), 1.0))
extern "C"
__global__ void convolution2D(float *I, const float* __restrict__ M, float *P, int channels, int width, int height) {
    __shared__ float N_ds[w][w];
    int k;
    for (k = 0; k < channels; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
                destY = dest / w, destX = dest % w,
                srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
                srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
                src = (srcY * width + srcX) * channels + k;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = I[src];
        else
            N_ds[destY][destX] = 0;
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w, destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
        src = (srcY * width + srcX) * channels + k;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = I[src];
            else
                N_ds[destY][destX] = 0;
        }
        __syncthreads();
        float accum = 0;
        int y, x;
        for (y = 0; y < Mask_width; y++)
            for (x = 0; x < Mask_width; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            P[(y * width + x) * channels + k] = clamp(accum);
        __syncthreads();
    }
}
extern "C"
__global__ void transposeConvolution2D(float *I, const float* __restrict__ M, float *P, int channels, int width, int height) {
    __shared__ float N_ds[w][w];
    int k;
    for (k = 0; k < channels; k++) {
        int dest = threadIdx.y * TILE_WIDTH + threadIdx.x,
                destY = dest / w, destX = dest % w,
                srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius,
                srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius,
                src = (srcY * width + srcX) * channels + k;
        if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
            N_ds[destY][destX] = I[src];
        else
            N_ds[destY][destX] = 0;
        dest = threadIdx.y * TILE_WIDTH + threadIdx.x + TILE_WIDTH * TILE_WIDTH;
        destY = dest / w, destX = dest % w;
        srcY = blockIdx.y * TILE_WIDTH + destY - Mask_radius;
        srcX = blockIdx.x * TILE_WIDTH + destX - Mask_radius;
        src = (srcY * width + srcX) * channels + k;
        if (destY < w) {
            if (srcY >= 0 && srcY < height && srcX >= 0 && srcX < width)
                N_ds[destY][destX] = I[src];
            else
                N_ds[destY][destX] = 0;
        }
        __syncthreads();
        float accum = 0;
        int y, x;
        for (y = 0; y < Mask_width; y++)
            for (x = 0; x < Mask_width; x++)
                accum += N_ds[threadIdx.y + y][threadIdx.x + x] * M[y * Mask_width + x];
        y = blockIdx.y * TILE_WIDTH + threadIdx.y;
        x = blockIdx.x * TILE_WIDTH + threadIdx.x;
        if (y < height && x < width)
            P[(y * width + x) * channels + k] = clamp(accum);
        __syncthreads();
    }
}
extern "C"
__global__ void Conv2D(float *input, const float* __restrict__ weight, float *data, int pad, int step, int i_row, int i_column, int w_row, int w_column, int w_depth, int d_row, int d_column) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < d_row && d < w_row) {
       float val = 0.0f;
       int x0, inputIndex, weightIndex, x = -pad + t * step, outputIndex = (t * d_column) + d;
       for (int j = 0; j < w_column; j++) {
           x0 = x + j;
           if (x0 < 0 || x0 >= i_row) {
               continue;
           }
           weightIndex = d * w_column * w_depth + j * w_depth;
           inputIndex = x0 * i_column;
           for (int c = 0; c < w_depth; c++, inputIndex++, weightIndex++) {
               val += input[inputIndex] * weight[weightIndex];
           }
       }
       data[outputIndex] = val;
   }
}
extern "C"
__global__ void TransposeConv2D(const float* __restrict__ input, const float* __restrict__ weight, float *data, int padding, int i_row, int i_column, int w_row, int w_column, int w_depth, int d_column) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < d_column && d < w_depth) {
        float val = 0.0f;
        int pad = w_column - 1 - padding;
        int sCore = w_column - 1;
        int x = -pad + t;
        int outputIndex = (t * d_column) + d;
        int sC, x0, weightIndex, inputIndex;
        for (int j = 0; j < w_column; j++) {
            x0 = x + j;
            if (x0 < 0 || x0 >= i_row) {
                continue;
            }
            sC = sCore - j;
            weightIndex = sC * w_depth + d;
            inputIndex = x0 * i_column;
            for (int c = 0; c < w_row; c++, inputIndex++) {
                val += input[inputIndex] * weight[c * w_depth * w_column + weightIndex];
            }
        }
        data[outputIndex] = val;
   }
}
extern "C"
__global__ void Convolution(const float* __restrict__ input, const float* __restrict__ error, float *data, int pad, int step, int i_row, int i_column, int r_row, int r_column, int d_row, int d_column, int d_depth) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    const int d = blockIdx.y * blockDim.y + threadIdx.y;
    if (t < r_row && d < d_row) {
        int x = -pad + t * step;
        int outputIndex = (t * r_column) + d;
        int y0, weightIndex, inputIndex;
        for (int j = 0; j < d_column; j++) {
            y0 = x + j;
            if (y0 < 0 || y0 >= i_row) {
                continue;
            }
            inputIndex = y0 * i_column;
            weightIndex = d * d_depth * d_column + j * d_depth;
            float tmp = 0;
            for (int c = 0; c < d_depth; c++, inputIndex++, weightIndex++) {
                tmp += input[inputIndex] * error[outputIndex];
            }
            data[weightIndex] = tmp;
        }
   }
}
extern "C"
__global__ void derSoftmax(const float* __restrict__ output, const float* __restrict__ error, float* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (k < row && i < column)
    {
       k = blockDim.x * blockIdx.x + threadIdx.x;
       i = blockDim.y * blockIdx.y + threadIdx.y;
       float value = 0.0f;
       int index = k * column;
       int indexI = index + i;
       data[indexI] = 0.0f;
       float o = output[indexI];
       float sum = 0.0f;
       for (int j = 0; j < column; j++) {
           int indexJ = index + j;
           if (i != j) {
               value = o * -output[indexJ];
           } else {
               value = o * (1.0f - o);
           }
           sum += error[indexJ] * value;
        }
        data[indexI] = sum;
    }
}
extern "C"
__global__ void derSoftmax_TYPE(const TYPE* __restrict__ output, const TYPE* __restrict__ error, TYPE* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (k < row && i < column)
    {
       k = blockDim.x * blockIdx.x + threadIdx.x;
       i = blockDim.y * blockIdx.y + threadIdx.y;
       TYPE value = sh[0];
       int index = k * column;
       int indexI = index + i;
       data[indexI] = sh[0];
       TYPE o = output[indexI];
       TYPE sum = sh[0];
       TYPE sh4 = sh[4];
       for (int j = 0; j < column; j++) {
           int indexJ = index + j;
           if (i != j) {
               value = o * -output[indexJ];
           } else {
               value = o * (sh4 - o);
           }
           sum = sum + error[indexJ] * value;
        }
        data[indexI] = sum;
    }
}
