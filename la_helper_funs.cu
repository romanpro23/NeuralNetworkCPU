#include <cuda_fp16.h>
#define TYPE half
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64
__device__ const int SharedMemorySize = 64 * 1024 / 2;
__device__ const int BLOCK_DIM = 32;
__device__ __constant__ half sh[14];
__inline__ __device__ half InfinityCheck(half v)
{
    int r = __hisinf(v);
    if (r == 1) {
        v = sh[12];
    }
    else if (r == -1) {
        v = -sh[12];
    }
    return v;
}
extern "C"
__global__ void fill(half* A, half alpha, int numElements)
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
__global__ void float2HalfVector(float* v, half* data, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        half d = __float2half_rn(v[i]);
        data[i] = InfinityCheck(d);
    }
}
extern "C"
__global__ void half2FloatVector(half* v, float* data, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        data[i] = __half2float(v[i]);
    }
}
extern "C"
__global__ void gelu(const float* __restrict__ A, float* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float a = A[i];
        float t = tanh(0.7978846f * a + 0.0356774f * (a * a * a));
        C[i] = 0.5f * a * (1.0f + t);
    }
}
extern "C"
__global__ void gelu_half(const half* __restrict__ A, half* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        half a = A[i];
        half t = __tanf(sh[1] * a + sh[2] * (a * a * a));
        C[i] = sh[3] * a * (sh[4] + t);
    }
}
extern "C"
__global__ void MatAdd(float* A, const float* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = A[k] + B[k];
    }
}
extern "C"
__global__ void MatAdd_half(half* A, const half* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = A[k] + B[k];
       A[k] = InfinityCheck(A[k]);
    }
}
extern "C"
__global__ void imageVector(const half* __restrict__ A, half* C, int rows, int columns, int depth, int sizeKernel)
{
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;
    const int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (h < rows && w < columns && z < sizeKernel)
    {
        int sizeKernel_X_depth = sizeKernel * depth;
        int sizeKernel_X_sizeKernel_X_depth_ = sizeKernel_X_depth * sizeKernel;
        int columns_X_sizeKernel_X_sizeKernel_X_depth = sizeKernel_X_sizeKernel_X_depth_ * columns / sizeKernel;
        int index = z * sizeKernel_X_depth + w / sizeKernel * sizeKernel_X_sizeKernel_X_depth_ + h / sizeKernel * columns_X_sizeKernel_X_sizeKernel_X_depth;
        for (int k = 0; k < sizeKernel; k++) {
            int indexInput = (h + z) * depth * columns + (w + k) * depth;
            for (int c = 0; c < depth; c++, index++, indexInput++) {
                C[index] = A[indexInput];
            }
        }
    }
}
extern "C"
__global__ void backImageVector(const half* __restrict__ A, half* C, int rows, int columns, int depth, int sizeKernel)
{
    const int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;
    const int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;
    const int z = blockDim.z * blockIdx.z + threadIdx.z;
    if (h < rows && w < columns && z < sizeKernel)
    {
        int sizeKernel_X_depth = sizeKernel * depth;
        int sizeKernel_X_sizeKernel_X_depth_ = sizeKernel_X_depth * sizeKernel;
        int columns_X_sizeKernel_X_sizeKernel_X_depth = sizeKernel_X_sizeKernel_X_depth_ * columns / sizeKernel;
        int index = z * sizeKernel_X_depth + w / sizeKernel * sizeKernel_X_sizeKernel_X_depth_ + h / sizeKernel * columns_X_sizeKernel_X_sizeKernel_X_depth;
        for (int k = 0; k < sizeKernel; k++) {
            int indexInput = (h + z) * depth * columns + (w + k) * depth;
            for (int c = 0; c < depth; c++, index++, indexInput++) {
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
__global__ void add3_half(const half* __restrict__ A, half* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    if (h < rows && w < columns) {
       int index = h * columns + w;
       C[index] = C[index] + A[w];
    }
}
extern "C"
__global__ void dot_VectorAndMatrix(float* C, const float* __restrict__ B, const float* __restrict__ A, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < rows) {
       float s = 0.0f;
       int index = h * columns;
       for (int j = 0; j < columns; j++, index++) {
           s += B[j] * A[index];
       }
       C[h] = s;
    }
}
extern "C"
__global__ void dot_VectorAndMatrix_half(half* C, const half* __restrict__ B, const half* __restrict__ A, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < rows) {
       half s = sh[0];
       for (int j = 0; j < columns; j++) {
           s += B[j] * A[h * columns + j];
       }
       C[h] = s;
    }
}
extern "C"
__global__ void MatMulKernel(TYPE *out, TYPE *in, TYPE *a, const int matrixHeight, const int matrixWidth) {
    // get variables for loop
    // copy section of b into shared mem
    // go through the threads vertically and sum them into a variable
    // atomic add these variables to the corresponding c index
    // looping is happening horizontally on the matrix
    // BLOCK_WIDTH is again horizontal
    // BLOCK_HEIGHT is going vertical
    // n / BLOCK_WIDTH blocks horizontally
    // m / BLOCK_HEIGHT block vertically
    // get variables for loop
    // variable for loop length: blockEltHeight
    __shared__ int blockElt;
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    if (threadIdx.x == 0) {
        if ((blockIdx.x + 1) * BLOCK_WIDTH <= matrixWidth)
            blockElt = BLOCK_WIDTH;
        else blockElt = matrixWidth % BLOCK_WIDTH;
        blockxInd = blockIdx.x * BLOCK_WIDTH;
        blockyInd = blockIdx.y * BLOCK_HEIGHT;
    }
    __syncthreads();
    // copy section of b into shared mem
    // use the first BLOCK_WIDTH of thread
    __shared__ TYPE b[BLOCK_WIDTH];
    if (threadIdx.x < blockElt)
        b[threadIdx.x] = in[blockxInd + threadIdx.x];
    __syncthreads();
    // summing variable
    TYPE cSum = (TYPE) sh[0];
    int threadyInd = blockyInd + threadIdx.x;
    // make sure we are inside the matrix verticallly
    if (threadyInd < matrixHeight) {
        // go through the threads vertically and sum them into a variable
        for (int i=0; i<blockElt; i++)
            // A col index   : blockIdx.x * BLOCK_WIDTH + i : blockxInd + i
            // A row index  : blockIdx.y * BLOCK_HEIGHT + threadIdx.x : blockyInd + threadIdx.x : threadyInd
            // B index : b[i]
            // cSum = B index * ( A col index * matrixHeight + A row index)
            cSum += b[i] * a[(blockxInd + i) * (matrixHeight) + (threadyInd)];
        // atomic add these variables to the corresponding c index
        atomicAdd(out + threadyInd, cSum);
    }
}
extern "C"
__global__ void MatMulKernelT(TYPE *out, TYPE *in, TYPE *a, const int matrixHeight, const int matrixWidth) {
    __shared__ int blockElt;
    __shared__ int blockxInd;
    __shared__ int blockyInd;
    if (threadIdx.x == 0) {
        if ((blockIdx.y + 1) * BLOCK_WIDTH <= matrixHeight)
            blockElt = BLOCK_WIDTH;
        else blockElt = matrixHeight % BLOCK_WIDTH;
        blockxInd = blockIdx.x * BLOCK_HEIGHT;
        blockyInd = blockIdx.y * BLOCK_WIDTH;
    }
    __syncthreads();
    // copy section of b into shared mem
    // use the first BLOCK_WIDTH of thread
    __shared__ TYPE b[BLOCK_WIDTH];
    if (threadIdx.x < blockElt)
        b[threadIdx.x] = in[blockyInd + threadIdx.x];
    __syncthreads();
    // summing variable
    TYPE cSum = (TYPE) sh[0];
    int threadxInd = blockxInd + threadIdx.x;
    // make sure we are inside the array horizontally
    if (threadxInd < matrixWidth) {
        // go through the threads vertically and sum them into a variable
        for (int i=0; i<blockElt; i++)
            // A col index : blockIdx.x * BLOCK_HEIGHT + threadIdx.x : blockxInd + threadIdx.x : threadxInd
            // A row index : blockIdx.y * BLOCK_WIDTH + i : blockyInd + i
            // B index : b[i]
            // cSum = B index * ( A col index * matrixHeight + A row index)
            cSum += b[i] * a[(threadxInd) * (matrixHeight) + (blockyInd + i)];
        // atomic add these variables to the corresponding c index
        atomicAdd(out + threadxInd , cSum);
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
            sum += A[i] * B[index];
       }
       C[j] = sum;
    }
}
extern "C"
__global__ void dotT_VectorAndMatrix_half(const half* __restrict__ A, const half* __restrict__ B, half* C, int rows, int columns)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < columns) {
       half sum = sh[0];
       for (int i = 0; i < rows; i++) {
            int index = i * columns + j;
            sum += A[i] * B[index];
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
__global__ void derivativeWeight_half(const half* __restrict__ input, const half* __restrict__ error, half* derWeight, int rows, int columns)
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
           d += matrix[index];
       }
       data[k] = d;
    }
  }
extern "C"
__global__ void addMatrix_half(const half* __restrict__ matrix, half* data, int width, int depth)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < depth) {
       half d = data[k];
       for (int i = 0; i < width; i++) { 
           int	index = i * depth + k;
           d += matrix[index];
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
__global__ void reverse(half* A, int rows, int columns, int depth)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int k = blockDim.z * blockIdx.z + threadIdx.z;
    const int index = i * blockDim.y * gridDim.y + j;
    if (index < rows * columns  && (k < depth))
    {
       const int index = rows - 1 - i;
       half valf = A[i * depth * columns + j * depth + k];
       half vals = A[index  * depth * columns + j * depth + k];
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
__global__ void sharedMem_transpose_half(half* R, half* M, int rows, int cols){
    __shared__ half M_Shared[BLOCK_DIM][BLOCK_DIM];
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
__global__ void matrixTransposeSolveBankConflicts(half* d_b, const half* __restrict__ d_a, int rows, int cols)
{
    __shared__ half mat[BLOCK_DIM][BLOCK_DIM + 1];
    int bx = blockIdx.x * BLOCK_DIM;
    int by = blockIdx.y * BLOCK_DIM;
    int i = by + threadIdx.y; int j = bx + threadIdx.x;
    int ti = bx + threadIdx.y; int tj = by + threadIdx.x;
    if (i<rows && j<cols)
       mat[threadIdx.y][threadIdx.x] = d_a[i*cols+j];
    __syncthreads();
    if (tj < cols && ti<rows)
       d_b[ti*rows+tj]=mat[threadIdx.x][threadIdx.y];
}
extern "C"
__global__ void transposeV3(half* AT, const half*  __restrict__ A, int width, int height){
  const int idx = threadIdx.x + blockDim.x*blockIdx.x;
  const int idy = threadIdx.y + blockDim.y*blockIdx.y;
  __shared__ half s_A[BLOCK_DIM][BLOCK_DIM+1];
  if(idx<width && idy<height){
     s_A[threadIdx.y][threadIdx.x] = A[idx + idy * height];
  }
  __syncthreads();
  const int idxT = threadIdx.x + blockDim.y*blockIdx.y;
  const int idyT = threadIdx.y + blockDim.x*blockIdx.x;
        if(idxT < width && idyT < height){
            AT[idxT + idyT * width] = s_A[threadIdx.x][threadIdx.y];
        }
    }
extern "C"
__global__ void matrixDiv(half* A, half B, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] /= B;
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
__global__ void addCopy_half(const half* __restrict__ matrix, half* data, int row, int col, int m_col, int start) 
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
           mean += input[index];
        }
        mean = mean / depth;
        P[1][x][y] = mean;
        float var = P[2][x][y];
        float sub;
        index = y * depth;
        mean = P[1][x][y];
        for (int k = 0; k < depth; k++, index++) {
            sub = input[index] - mean;
            var += sub * sub;
        }
        var = var / depth;
        P[2][x][y] = var;
        float varSqrt = sqrtf(var + 0.0000001f);
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
        float dVar_m = -0.5f * powf(var + 0.0000001f, -1.5f);
        int index = y * depth;
        float derVariance = 0.0f;
        for (int k = 0; k < depth; k++, index++) {
            derVariance += errorNL[index] * gamma[k] * (input[index] - mean);
        }
        derVariance *= dVar_m;
        dVar_m = 0.0f;
        float derMean = 0.0f;
        float dMean = -1.0f / sqrtf(var + 0.0000001f);
        index = y * depth;
        for (int k = 0; k < depth; k++, index++) {
            derMean += errorNL[index] * gamma[k];
            dVar_m += input[index] - mean;
        }
        derMean *= dMean;
        derMean += (-2.0f * derVariance * dVar_m) / depth;
        derMean /= depth;
        derVariance *= 2.0f / (depth);
        float _dVar = 1.0f / sqrtf(var + 0.0000001f);
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
__global__ void dropout(half* A, half* random, half chanceDrop, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       half drop = sh[4] / (sh[4] - chanceDrop);
       if (random[idx] > chanceDrop)
       {
           A[idx] = A[idx] * drop;
       }
    }
}
extern "C"
__global__ void sub_gpu_half2(const half2* __restrict__ first, const half2* __restrict__ second, half2* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = __hsub2(first[idx], second[idx]);
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
__global__ void sub_gpu_half(const half* __restrict__ first, const half* __restrict__ second, half* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = first[idx] - second[idx];
    }
}
extern "C"
__global__ void sub_half_float_gpu(const half* __restrict__ first, const float* __restrict__ second, float* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = __half2float(first[idx]) - second[idx];
    }
}
extern "C"
__global__ void sub_float_half_gpu(const float* __restrict__ first, const half* __restrict__ second, float* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = first[idx] - __half2float(second[idx]);
    }
}
extern "C"
__global__ void mul(float* result, float val, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] *= val;
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
__global__ void clip_half(half* data, half min, half max, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        half a = data[idx];
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
__global__ void pow2_half(half* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] *= data[idx];
    }
}
extern "C"
__global__ void subAbs_half(half* first, half* second, half* result, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       result[i] = __habs(first[i] - second[i]);
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
__global__ void sum_half(half* data, half* result, int numElements)
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
       result[i] = diff / fabsf(diff) + 0.0000001f;
    }
}
extern "C"
__global__ void fisnan(const half* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (__hisnan(data[idx])) {
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
__global__ void hisinf(const half* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (__hisinf(data[idx])) {
               result[0] = idx;
           }
           if (__hisinf(data[idx]) == -1) {
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
__global__ void momentum(half* data, const half* __restrict__ array, half decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + array[idx] * (sh[4] - decay);
    }
}
extern "C"
__global__ void momentum_float(float* data, const float* __restrict__ array, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + array[idx] * (1.0f - decay);
    }
}
extern "C"
__global__ void momentumPow2(half* data, const half* __restrict__ vector, half decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + (sh[4] - decay) * vector[idx] * vector[idx];
    }
}
extern "C"
__global__ void momentumPow2_float(float* data, const float* __restrict__ vector, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + (1.0f - decay) * vector[idx] * vector[idx];
    }
}
extern "C"
__global__ void subDivSqrtNorm_half(const half* __restrict__ nominator, const half* __restrict__ denominator, half lr, half normN, half normD, half* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       half sh5 = sh[5];
       half cur_lr = lr / (normN +  sh5);
       atomicAdd(&data[idx], -(cur_lr * (nominator[idx]) / (hsqrt(denominator[idx] / normD) + sh5)));
    }
}
extern "C"
__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       float cur_lr = lr / (normN + 0.0000001f);
       atomicAdd(&data[idx], -(cur_lr * (nominator[idx]) / (sqrtf(denominator[idx] / normD) + 0.000001f)));
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
__global__ void addBackCopy_half(const half* __restrict__ matrix, int m_column, int row, int column, int start, half* data)
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
__global__ void dropoutBack(const half* __restrict__ output, const half* __restrict__ error, half drop, half* data, int numElements)
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
__global__ void mask_half(const half* __restrict__ A, half val, half newVal, half* C, int numElements)
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
__global__ void fillUnderDiagonal_half(int column, half val, half* data, int numElements)
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
        float val = tanh(0.7978846f * x + 0.0356774f * x * x * x);
        data[idx] = error[idx] * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f * x * x));
    }
}
extern "C"
__global__ void derGelu_half(const half* __restrict__ input, const half* __restrict__ error, half* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        half x = input[idx];
        half val = tanh(sh[1] * x + sh[2] * x * x * x);
        data[idx] = error[idx] * sh[3] * (sh[4] + val + x * (sh[4] - val * val) * (sh[10] + sh[11] * x * x));
    }
}
__device__ static __forceinline__ float _shfl_up(float var, unsigned int delta, int width=32, unsigned mask=0xffffffff)
{
#if ( __CUDA_ARCH__ >= 300)
#if (__CUDACC_VER_MAJOR__ >= 9)
   var = __shfl_up_sync(mask, var, delta, width);
#else
   var = __shfl_up(var, delta, width);
#endif
#endif
return var;
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
        y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
    }
        __syncthreads();
    }
    if (tid < nRows) dy[tid] = y_val;
}
extern "C"
__global__ void matvec_kernel_half(const half* __restrict__ dA, const half* __restrict__ dx, half* __restrict__ dy, const unsigned int nRows, const unsigned int nCols)
{
    const unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
    __shared__ half x_shared[BLOCK_SIZE];
    half y_val = 0.0;
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
        y_val += dA[tid * nCols + (e + BLOCK_SIZE * m)] * x_shared[e];
    }
        __syncthreads();
    }
    if (tid < nRows) dy[tid] = y_val;
}
extern "C"
__global__ void Softmax(const float* __restrict__ input, float* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int g = blockDim.y * blockIdx.y + threadIdx.y;
    if (k < row && g < column)
    {
       __shared__ float max[512];
       __shared__ float sum[512];
       int index = k * column + g;
       float inx = input[index];
       max[k] = inx;
       sum[k] = 0.0f;
       __syncthreads();
       if (max[k] < inx)
           max[k] = inx;
       __syncthreads();
       float d = __expf(inx - max[k]);
       atomicAdd(&sum[k], d);
       data[index] = d;
       __syncthreads();
       if (sum[k] != 0.0f) {
           data[index] /= sum[k];
       }        else {
           data[index] /= 0.0000001f;
       }     }
}
extern "C"
__global__ void Softmax_half(const half* __restrict__ input, half* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int g = blockDim.y * blockIdx.y + threadIdx.y;
    if (k < row && g < column)
    {
       __shared__ half max[512];
       __shared__ half sum[512];
       int index = k * column + g;
       half inx = input[index];
       max[k] = inx;
       sum[k] = sh[0];
       __syncthreads();
       if (max[k] < inx)
           max[k] = inx;
       __syncthreads();
       half d = __expf(inx - max[k]);
       atomicAdd(&sum[k], d);
       data[index] = d;
       __syncthreads();
       if (sum[k] != sh[0]) {
           data[index] /= sum[k];
       }        else {
           data[index] /= 0.0000001f;
       }     }
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
__global__ void derSoftmax_half(const half* __restrict__ output, const half* __restrict__ error, half* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    if (k < row && i < column)
    {
       k = blockDim.x * blockIdx.x + threadIdx.x;
       i = blockDim.y * blockIdx.y + threadIdx.y;
       half value = sh[0];
       int index = k * column;
       int indexI = index + i;
       data[indexI] = sh[0];
       half o = output[indexI];
       half sum = sh[0];
       for (int j = 0; j < column; j++) {
           int indexJ = index + j;
           if (i != j) {
               value = o * -output[indexJ];
           } else {
               value = o * (sh[4] - o);
           }
             sum += error[indexJ] * value;
        }
        sum = InfinityCheck(sum);
        data[indexI] = sum;
    }
}
