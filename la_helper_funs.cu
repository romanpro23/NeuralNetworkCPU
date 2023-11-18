#include <cuda_fp16.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#define TYPE half
#define BLOCK_HEIGHT 1024
#define BLOCK_WIDTH 64
__device__ const int SharedMemorySize = 64 * 1024 / 2;
__device__ const int BLOCK_DIM = 32;
__device__ __constant__ half sh[13];
extern "C"
__global__ void fill(half* A, half alpha, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = alpha;
    }
}
extern "C"
__global__ void gelu(const half* __restrict__ A, half* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        half a = A[i];
        half t = tanh(sh[1] * a + sh[2] * (a * a * a));
        C[i] = sh[3] * a * (sh[4] + t);
    }
}
extern "C"
__global__ void set(half* A, int i, half alpha)
{
    A[i] = alpha;
}
extern "C"
__global__ void MatAdd(half2* A, const half2* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       A[k] = __hadd2(A[k], B[k]);
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
__global__ void add3(const half* __restrict__ A, half* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    if (h < rows && w < columns) {
       int index = h * blockDim.y * gridDim.y + w;
          C[index] += A[w];
    }
}
extern "C"
__global__ void dot_VectorAndMatrix(TYPE* C, const TYPE* __restrict__ B, const TYPE* __restrict__ A, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < rows) {
       TYPE s = sh[0];
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
__global__ void dotT_VectorAndMatrix(const half* __restrict__ A, const half* __restrict__ B, half* C, int rows, int columns)
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
__global__ void derivativeWeight(const half* __restrict__ input, const half* __restrict__ error, half* derWeight, int rows, int columns)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < rows && j < columns) {
       const int index = i * columns + j;
       half m = error[i] * input[j];
       half v = derWeight[index] + m;
       derWeight[index] = v;
    }
}
extern "C"
__global__ void findMean_part(const half* __restrict__ A, half* C, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
       half s = sh[0];
       int index = j * depth;
       for (int k = 0; k < depth; k++, index++) {
           s += A[index];
       }
       C[j] = s;
    }
}
extern "C"
__global__ void generateErrorNorm(const half* __restrict__ errorNL, const half* __restrict__ gamma, half* errorNorm, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int k = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = j * depth + k;
    if (j < width && k < depth) {
       errorNorm[index] = errorNL[index] * gamma[k];
    }
}
extern "C"
__global__ void derVar_part_2(const half* __restrict__ error, const half* __restrict__ input, const half* __restrict__ mean, half* derVariance, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        int index = j * depth;
        half m = mean[j];
        half s = sh[0];
        for (int k = 0; k < depth; k++, index++) {
           s += error[index] * (input[index] - m);
        }
        derVariance[j] = s;
    }
}
extern "C"
__global__ void derMean_part_2(const half* __restrict__ error, const half* __restrict__ input, const half* __restrict__ mean, half* derMean, half* dVar, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        half DM = sh[0];
        half DV = DM;
        half m = mean[j];
        for (int k = 0; k < depth; k++) {
           int index = j * depth + k;
           DM += error[index];
           DV += input[index] - m;
        }
        derMean[j] = DM;
        dVar[j] = DV;
    }
}
extern "C"
__global__ void derNorm_part_2(const half* __restrict__ errors, const half* __restrict__ dVar, const half* __restrict__ errorVar, const half* __restrict__ input, const half* __restrict__ mean, const half* __restrict__ errorMean, half* error, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int k = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = j * depth + k;
    if (j < width && k < depth) {
        error[index] = errors[index] * dVar[j] + errorVar[j] * (input[index] - mean[j]) + errorMean[j];
    }
}
extern "C"
__global__ void derivativeWeight_2(const half* __restrict__ error, const half* __restrict__ output, const half* __restrict__ betta, const half* __restrict__ gamma, half* derBetta, half* derGamma, int width, int depth)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < depth) {
       half dB = derBetta[i];
       half dG = derGamma[i];
       half g = gamma[i];
       half b = betta[i];
       for (int j = 0; j < width; j++) { 
           int ind = j * depth + i;
           dB += error[ind];
           dG += error[ind] * ((output[ind] - b) / g);
       }
       derBetta[i] = dB;
       derGamma[i] = dG;
    }
}
extern "C"
__global__ void findVariance_part(const half* __restrict__ input, const half* __restrict__ mean, half* var, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        half s = sh[0];
        half m = mean[j];
        int index = j * depth;
        for (int k = 0; k < depth; k++, index++) {
           half sub = input[index] - m;
           s += sub * sub;
        }
        var[j] = s;
    }
}
extern "C"
__global__ void normalization_part_2(const half* __restrict__ input, const half* __restrict__ mean, const half* __restrict__ varSqrt, half* normOutput, const half* __restrict__ gamma, const half* __restrict__ betta, half* output, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    if (j < width && k < depth) {
        int index = j * depth + k;
        half nO = (input[index] - mean[j]) / varSqrt[j];
        output[index] = nO * gamma[k] + betta[k];
        normOutput[index] = nO;
    }
}
extern "C"
__global__ void addMatrix(const half* __restrict__ matrix, half* data, int width, int depth)
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
__global__ void set2(half* A, int i, int j, int k, int columns, int depth, half value)
{
    A[i * depth * columns + j * depth + k] = value;
}
extern "C"
__global__ void transpose_naive(half* odata, const half* __restrict__ idata, int width, int height)
{
    unsigned int xIndex = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int yIndex = blockDim.y * blockIdx.y + threadIdx.y;
    if (xIndex < width && yIndex < height)
    {
        unsigned int index_in  = xIndex + width * yIndex;
        unsigned int index_out = yIndex + height * xIndex;
        odata[index_out] = idata[index_in];
    }
}
extern "C"
__global__ void sharedMem_transpose(half* R, half* M, int rows, int cols){
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
__global__ void addCopy(const half* __restrict__ matrix, half* data, int row, int col, int m_col, int start) 
{
    const int x = blockDim.x * blockIdx.x + threadIdx.x;
    const int y = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = x * blockDim.y * gridDim.y + y;
    if (index < row * m_col)
    {
        const int indexIn = x * col + start * m_col + y;
        data[indexIn] = matrix[index];
    }
}
extern "C"
__global__ void normalization_part_1(half* A, const half* __restrict__ var, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       A[i] = hsqrt(var[i] + sh[5]);
    }
}
extern "C"
__global__ void DenseLayerForward(half*** P, half* weight, const half* __restrict__ threshold, int row, int column, int n)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < row) {
       int index = y * column;
       half sum = P[1][x][y];
       for (int j = 0; j < column; j++, index++) {
           P[1][x][y] += P[0][x][j] * weight[index];
       }
       P[1][x][y] = sum;
       atomicAdd(&P[1][x][y], threshold[y]);
    }
}
extern "C"
__global__ void dot(half* const __restrict__ a, const half* __restrict__ b, half* result, int row1, int col1, int row2, int col2)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < row1 && j < col2) {
       half sum = sh[0];
       for (int k = 0; k < col1; k++) {
           sum = sum + a[i * col1 + k] * b[k * col2 + j];
       }
       result[i * col2 + j] = sum;
    }
}
__device__ size_t getGlobalIdx_3D_3D()
{
    size_t blockId = blockIdx.x + blockIdx.y * gridDim.x
            + gridDim.x * gridDim.y * blockIdx.z;
    size_t threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
            + (threadIdx.z * (blockDim.x * blockDim.y))
            + (threadIdx.y * blockDim.x) + threadIdx.x;
    return threadId;
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
__global__ void sub_gpu(const half2* __restrict__ first, const half2* __restrict__ second, half2* result, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] = __hsub2(first[idx], second[idx]);
    }
}
extern "C"
__global__ void mul(half* result, half val, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] *= val;
    }
}
extern "C"
__global__ void clip(half* data, half min, half max, int numElements)
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
__global__ void indexMaxElement(half* data, half* max_value, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        if (data[idx] > *max_value) {
            *max_value = data[idx];
            *result = idx;
        }
    }
}
extern "C"
__global__ void derVar_part_1(const half* __restrict__ var, half* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dVar[idx] = sh[6] * hexp2(sh[7] * hlog2(var[idx] + sh[5]));
    }
}
extern "C"
__global__ void derVar_part_3(const half* __restrict__ dVar, half* derVariance, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       derVariance[idx] *= dVar[idx];
    }
}
extern "C"
__global__ void derMean_part_1(const half* __restrict__ var, half* dMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dMean[idx] = (sh[8] / hsqrt(var[idx] + sh[5]));
    }
}
extern "C"
__global__ void derMean_part_3(const half* __restrict__ dMean, const half* __restrict__ derVar, const half* __restrict__ dVar, int depth, half* derMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       half dM = derMean[idx];
       dM *= dMean[idx];
       dM += (sh[9] * derVar[idx] * dVar[idx]) / (__int2half_rn(depth));
       derMean[idx] = dM;
    }
}
extern "C"
__global__ void derNorm_part_1(const half* __restrict__ var, half* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dVar[idx] = sh[4] / hsqrt(var[idx] + sh[5]);
    }
}
extern "C"
__global__ void pow2(half* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       half d = data[idx];
       data[idx] = d * d;
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
__global__ void momentum(half* data, const half* __restrict__ array, half decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + array[idx] * (sh[4] - decay);
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
__global__ void subDivSqrtNorm(const half* __restrict__ nominator, const half* __restrict__ denominator, half lr, half normN, half normD, half* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       half sh5 = sh[5];
       half cur_lr = lr / (normN +  sh5);
       data[idx] -= (half)(cur_lr * (nominator[idx]) / (hsqrt(denominator[idx] / normD) +  sh5));
    }
}
extern "C"
__global__ void addBackCopy(const half* __restrict__ matrix, int m_column, int row, int column, int start, half* data)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    const int index = x * blockDim.y * gridDim.y + y;
    if (index < row * column) {
       const int indexOut = x * m_column + start * column + y;
       data[index] = matrix[indexOut];
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
__global__ void mask(const half* __restrict__ A, half val, half newVal, half* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       if (A[i] == val) {
           C[i] = newVal;
       }
    }
}
extern "C"
__global__ void fillUnderDiagonal(int column, half val, half* data, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        for (int j = 0; j < i + 1; j++) {
            data[i * column + j] = val;
        }
    }
}
extern "C"
__global__ void derGelu(const half* __restrict__ input, const half* __restrict__ error, half* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        half x = input[idx];
        half val = tanh(sh[1] * x + sh[2] * (x * x * x));
        data[idx] = error[idx] * sh[3] * (sh[4] + val + x * (sh[4] - val * val) * (sh[10] + sh[11] * x * x));
    }
}
extern "C"
__global__ void matrixMultiplicationKernel(const half* __restrict__ A, const half* __restrict__ B, half* C, int width, int P, int Q) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = r * blockDim.y * gridDim.y + c;
    if (index < P * Q) {
        half value = __float2half(0.0f);
        for(int k = 0; k < width; k++) {
            value += A[r * width + k] * B[k * Q + c];
        }
        C[r * Q + c] = value;
    }
}
extern "C"
__global__ void Softmax(const half* __restrict__ input, half* data, int column, int numElements)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       half sum = sh[0];
       int index = k * column;
       half max = input[index];
       for (int i = 1; i < column; i++, index++) {
           half inx = input[index];
           if (max < inx)
               max = inx;
       }
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           half d = hexp(input[index] - max);
           sum += d;
           data[index] = d;
       }
       if (sum == sh[0]) {
           sum = sh[5];
       }
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           data[index] /= sum;
       }
    }
}
extern "C"
__global__ void derSoftmax(const half* __restrict__ output, const half* __restrict__ error, half* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = k * blockDim.y * gridDim.y + i;
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
        data[indexI] = sum;
    }
}
