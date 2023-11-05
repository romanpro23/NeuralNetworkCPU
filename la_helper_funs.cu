const __device__ float epsilon = 0.001f;
#define MAX_FLOAT_EXP 		80
#include <cuda_fp16.h>
__device__ int SharedMemorySize = 64 * 1024 / 4;
__device__ const int BLOCK_DIM = 32;
extern "C"
__global__ void fill(float* A, float alpha, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = alpha;
    }
}
extern "C"
__global__ void gelu(const float* __restrict__ A, float* C, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        float a = A[i];
        C[i] = 0.5f * a * (1.0f + tanh(0.7978846f * a + 0.0356774f * (a * a * a)));
    }
}
extern "C"
__global__ void set(float* A, int i, float alpha)
{
    A[i] = alpha;
}
extern "C"
__global__ void multiplyIndex(float* A, int alpha, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = i * alpha;
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
__global__ void backImageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)
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
       int index = h * blockDim.y * gridDim.y + w;
       w = blockDim.y * blockIdx.y + threadIdx.y;
          C[index] += A[w];
    }
}
extern "C"
__global__ void dot_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < rows) {
       float s = 0.0f;
       for (int j = 0; j < columns; j++) {
           s += B[j] * A[h * columns + j];
       }
       C[h] = s;
    }
}
extern "C"
__global__ void dotT_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < columns) {
       float sum = 0.0f;
       for (int i = 0; i < rows; i++) {
            int index = floorf(i * columns + j);
            sum += A[i] * B[index];
       }
       C[j] = sum;
    }
}
extern "C"
__global__ void derivativeWeight(const float* __restrict__ input, const float* __restrict__ error, float* derWeight, int rows, int columns)
{
    unsigned idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < rows * columns) {
        derWeight[idx] += error[idx / columns] * input[idx % columns];
    }
}
extern "C"
__global__ void findMean_part(const float* __restrict__ A, float* C, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
       float s = 0.0f;
       int index = j * depth;
       for (int k = 0; k < depth; k++, index++) {
           s += A[index];
       }
       C[j] = s;
    }
}
extern "C"
__global__ void generateErrorNorm(const float* __restrict__ errorNL, const float* __restrict__ gamma, float* errorNorm, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int k = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = j * blockDim.y * gridDim.y + k;
    if (index < width * depth) {
       errorNorm[index] = errorNL[index] * gamma[k];
    }
}
extern "C"
__global__ void derVar_part_2(const float* __restrict__ error, const float* __restrict__ input, const float* __restrict__ mean, float* derVariance, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        int index = j * depth;
        float m = mean[j];
        float s = 0.0f;
        for (int k = 0; k < depth; k++, index++) {
           s += (float)(error[index] * (input[index] - m));
        }
        derVariance[j] = s;
    }
}
extern "C"
__global__ void derMean_part_2(const float* __restrict__ error, const float* __restrict__ input, const float* __restrict__ mean, float* derMean, float* dVar, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        int index = j * depth;
        float DM = 0.0f;
        float DV = 0.0f;
        float m = mean[j];
        for (int k = 0; k < depth; k++, index++) {
           DM += error[index];
           DV += input[index] - m;
        }
        derMean[j] = DM;
        dVar[j] = DV;
    }
}
extern "C"
__global__ void derNorm_part_2(const float* __restrict__ errors, const float* __restrict__ dVar, const float* __restrict__ errorVar, const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ errorMean, float* error, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int k = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = j * blockDim.y * gridDim.y + k;
    if (index < width * depth) {
        error[index] = errors[index] * dVar[j] + errorVar[j] * (input[index] - mean[j]) + errorMean[j];
    }
}
extern "C"
__global__ void derivativeWeight_2(const float* __restrict__ error, const float* __restrict__ output, const float* __restrict__ betta, const float* __restrict__ gamma, float* derBetta, float* derGamma, int width, int depth)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < depth) {
       float dB = derBetta[i];
       float dG = derGamma[i];
       float g = gamma[i];
       float b = betta[i];
       for (int j = 0; j < width; j++) { 
           int ind = floorf(j * depth + i);
           dB += error[ind];
           dG += error[ind] * ((output[ind] - b) / g);
       }
       derBetta[i] = dB;
       derGamma[i] = dG;
    }
}
extern "C"
__global__ void findVariance_part(const float* __restrict__ input, const float* __restrict__ mean, float* var, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        float s = 0.0f;
        float m = mean[j];
        int index = j * depth;
        for (int k = 0; k < depth; k++, index++) {
           float sub = input[index] - m;
           s += sub * sub;
        }
        var[j] = s;
    }
}
extern "C"
__global__ void normalization_part_2(const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ varSqrt, float* normOutput, const float* __restrict__ gamma, const float* __restrict__ betta, float* output, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    if (j < width && k < depth) {
        int index = j * blockDim.y * gridDim.y + k;
        float nO = (input[index] - mean[j]) / varSqrt[j];
        output[index] = nO * gamma[k] + betta[k];
        normOutput[index] = nO;
    }
}
extern "C"
__global__ void add_NNMatrix(const float* __restrict__ matrix, float* data, int width, int depth)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < depth) {
       double d = (double)data[k];
       for (int i = 0; i < width; i++) { 
           int	index = floorf(i * depth + k);
           d += matrix[index];
       }
       data[k] = (float)d;
    }
  }
extern "C"
__global__ void reverse(float* A, int rows, int columns, int depth)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    const int j = blockDim.y * blockIdx.y + threadIdx.y;
    const int k = blockDim.z * blockIdx.z + threadIdx.z;
    const int index = i * blockDim.y * gridDim.y + j;
    if (index < rows * columns  && (k < depth))
    {
       const int index = rows - 1 - i;
       float valf = A[i * depth * columns + j * depth + k];
       float vals = A[index  * depth * columns + j * depth + k];
       A[i  * depth * columns + j * depth + k] = valf;
       A[index  * depth * columns + j * depth + k] = vals;
    }
}
extern "C"
__global__ void set2(float* A, int i, int j, int k, int columns, int depth, float value)
{
    A[i * depth * columns + j * depth + k] = value;
}
extern "C"
__global__ void MatAdd(float* A, const float* __restrict__ B, int numElements)
{
    const int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements) {
       atomicAdd(&A[k], B[k]);
    }
}
extern "C"
__global__ void matrixDiv(float* A, float B, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] /= B;
    }
}
extern "C"
__global__ void vectorScalarSet(float* A, float alpha, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        A[i] = alpha;
    }
}
extern "C"
__global__ void addCopy(const float* __restrict__ matrix, float* data, int row, int col, int m_col, int start) 
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
__global__ void normalization_part_1(float* A, const float* __restrict__ var, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       A[i] = (float)sqrt(var[i] + epsilon);
    }
}
extern "C"
__global__ void NormalizationLayerForward2D(float*** P, const float* __restrict__ gamma, const float* __restrict__ betta, int width, int depth, int n)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < width) {
       float mean = P[3][x][y];
       for (int index = y * depth, int k = 0; k < depth; k++, index++) {
           mean += P[2][x][index];
       }
       P[3][x][y] = mean;
       float var = P[4][x][y];
       float sub;
       for (int index = y * depth, int k = 0; k < depth; k++, index++) {
           sub = (float)(P[2][x][index] - mean);
           var += sub * sub;
       }
       var = var / depth;
       P[4][x][y] = var;
       float varSqrt = (float) (sqrt(var + epsilon));
       for (int index = y * depth, int k = 0; k < depth; k++, index++) {
           float nO = ((float)(P[2][x][index] - mean)) / varSqrt;
           P[0][x][index] = nO;
           P[1][x][index] = nO * gamma[k] + betta[k];
       }
    }
}
extern "C"
__global__ void DenseLayerForward(float*** P, float* weight, const float* __restrict__ threshold, int row, int column, int n)
{
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    if (x < n && y < row) {
       int index = y * column;
       float sum = P[1][x][y];
       for (int j = 0; j < column; j++, index++) {
           P[1][x][y] += P[0][x][j] * weight[index];
       }
       P[1][x][y] = sum;
       atomicAdd(&P[1][x][y], threshold[y]);
    }
}
extern "C"
__global__ void dot(float* const __restrict__ a, const float* __restrict__ b, float* result, int row1, int col1, int row2, int col2)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < row1 && j < col2) {
       float sum = 0.0f;
       for (int k = 0; k < col1; k++) {
           sum = sum + a[i * col1 + k] * b[k * col2 + j];
       }
       result[i * col2 + j] = sum;
    }
}
extern "C"
__global__ void transpose(float* odata, const float* __restrict__ idata, int width, int height)
{
    __shared__ float block[BLOCK_DIM][BLOCK_DIM+1];
    unsigned int xIndex = blockIdx.x * BLOCK_DIM + threadIdx.x;
    unsigned int yIndex = blockIdx.y * BLOCK_DIM + threadIdx.y;
    if((xIndex < width) && (yIndex < height))
    {
        unsigned int index_in = yIndex * width + xIndex;
        block[threadIdx.y][threadIdx.x] = idata[index_in];
    }
    __syncthreads();
    xIndex = blockIdx.y * BLOCK_DIM + threadIdx.x;
    yIndex = blockIdx.x * BLOCK_DIM + threadIdx.y;
    if((xIndex < height) && (yIndex < width))
    {
        unsigned int index_out = yIndex * height + xIndex;
        odata[index_out] = block[threadIdx.x][threadIdx.y];
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
__global__ void dropout(float* A, float* random, double chanceDrop, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       float drop = (float) (1.0f / (1.0f - chanceDrop));
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
__global__ void mul(float* result, float val, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       result[idx] *= val;
    }
}
extern "C"
__global__ void indexMaxElement(float* data, float* max_value, int* result, int numElements)
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
__global__ void derVar_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dVar[idx] = (float) (-0.5f * pow((var[idx] + epsilon), -1.5f));
    }
}
extern "C"
__global__ void derVar_part_3(const float* __restrict__ dVar, float* derVariance, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       derVariance[idx] *= dVar[idx];
    }
}
extern "C"
__global__ void derMean_part_1(const float* __restrict__ var, float epsilon, float* dMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dMean[idx] = (float) (-1.0f / sqrt(var[idx] + epsilon));
    }
}
extern "C"
__global__ void derMean_part_3(const float* __restrict__ dMean, const float* __restrict__ derVar, const float* __restrict__ dVar, int depth, float* derMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       float dM = derMean[idx];
       dM *= dMean[idx];
       dM += (float) ((-2.0f * derVar[idx] * dVar[idx]) / (depth));
       derMean[idx] = dM;
    }
}
extern "C"
__global__ void derNorm_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dVar[idx] = (float) (1.0f / sqrt(var[idx] + epsilon));
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
__global__ void sum(const float* __restrict__ data, float* sum, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       atomicAdd(sum, data[idx]);
    }
}
extern "C"
__global__ void isnan(const float* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (result[0] == 0) {
           if (isnan(data[idx])) {
               result[0] = 1;
           }
       }
    }
}
extern "C"
__global__ void divide_add(float* A, const float* B, const int C)
{
   atomicAdd(A, B[0] / C);
}
extern "C"
__global__ void momentum(float* data, const float* __restrict__ array, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + array[idx] * (1.0f - decay);
    }
}
extern "C"
__global__ void momentumPow2(float* data, const float* __restrict__ vector, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = decay * data[idx] + (1.0f - decay) * vector[idx] * vector[idx];
    }
}
extern "C"
__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        atomicAdd(&data[idx], -(lr / (normN + 0.0000001f)) * (nominator[idx]) / (sqrt(denominator[idx] / normD) + 0.0000001f));
    }
}
extern "C"
__global__ void addBackCopy(const float* __restrict__ matrix, int m_column, int row, int column, int start, float* data)
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
__global__ void dropoutBack(const float* __restrict__ output, const float* __restrict__ error, float drop, float* data, int numElements)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       if (output[idx] != 0) {
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
__global__ void derGelu(const float* __restrict__ input, const float* __restrict__ error, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
        float x = input[idx];
        float val = tanh(0.7978846f * x + 0.0356774f * (x * x * x));
        data[idx] = (float)(error[idx] * 0.5f * (1.0f + val + x * (1.0f - val * val) * (0.79788846f + 0.1070322f * x * x)));
    }
}
extern "C"
__global__ void matrixMultiplicationKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int width, int P, int Q) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    const int index = r * blockDim.y * gridDim.y + c;
    if (index < P * Q) {
        float value = (float)(0.0);
        for(int k = 0; k < width; k++) {
            value += A[r * width + k] * B[k * Q + c];
        }
        C[r * Q + c] = value;
    }
}
extern "C"
__global__ void Softmax(const float* __restrict__ input, float* data, int column, int numElements)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       float sum = 0;
       int index = k * column;
       float max = input[index];
       for (int i = 1; i < column; i++, index++) {
           if (max < input[index])
               max = input[index];
       }
       index = k * column;
       float E = 2.718281828459045f;
       for (int i = 0; i < column; i++, index++) {
           data[index] = (float)(pow(E, input[index] - max));
               sum += data[index];
       }
       sum += 0.00000001f;
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           data[index] /= sum;
       }
    }
}
extern "C"
__global__ void derSoftmax(const float* __restrict__ output, const float* __restrict__ error, float* data, int row, int column)
{
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int idx = k * blockDim.y * gridDim.y + i;
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
