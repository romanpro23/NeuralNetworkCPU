extern "C"
__global__ void fill(float* A, float alpha, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        A[i] = alpha;
    }
}
extern "C"
__global__ void gelu(const float* __restrict__ A, float* C, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        C[i] = (float) (0.5f * A[i] * (1 + tanh((double)(0.7978846 * A[i] + 0.0356774 * pow((double)A[i], 3.0)))));
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
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        A[i] = i * alpha;
    }
}
extern "C"
__global__ void imageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)
{
    int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;
    int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;
    int index = h + w * blockDim.x * gridDim.x;
    __shared__ int indexInput;
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
       indexInput = 0;
    }
    if ((h < rows) && (w < columns))
    {
        for (int j = 0; j < sizeKernel; j++) {
            for (int k = 0; k < sizeKernel; k++) {
                indexInput = (h + j) * depth * columns + (w + k) * depth;
                for (int c = 0; c < depth; c++, index++, indexInput++) {
                     C[index] = A[indexInput];
                }
             }
         }
    }
}
extern "C"
__global__ void backImageVector(const float* __restrict__ A, float* C, int rows, int columns, int depth, int sizeKernel)
{
    int h = (blockDim.x * blockIdx.x + threadIdx.x) * sizeKernel;
    int w = (blockDim.y * blockIdx.y + threadIdx.y) * sizeKernel;
    int index = h + w * blockDim.x * gridDim.x;
    __shared__ int indexInput;
    if ((threadIdx.x == 0) && (threadIdx.y == 0)) {
       indexInput = 0;
    }
    if ((h < rows) && (w < columns))
    {
        for (int j = 0; j < sizeKernel; j++) {
            for (int k = 0; k < sizeKernel; k++) {
                indexInput = (h + j) * depth * columns + (w + k) * depth;
                for (int c = 0; c < depth; c++, index++, indexInput++) {
                     C[indexInput] = A[index];
                }
             }
         }
    }
}
extern "C"
__global__ void add3(const float* __restrict__ A, float* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    int index = h + w * blockDim.x * gridDim.x;
    if ((h < rows) && (w < columns))
    {
                     C[index] += A[w];
    }
}
extern "C"
__global__ void dot_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    int index = h + w * blockDim.x * gridDim.x;
    if ((h < rows) && (w < columns))
    {
          C[h] += A[w] * B[index];
    }
}
extern "C"
__global__ void dotT_VectorAndMatrix(const float* __restrict__ data, const float* __restrict__ matrix, float* result, int rows, int columns)
{
    int h = blockDim.x * blockIdx.x + threadIdx.x;
    int w = blockDim.y * blockIdx.y + threadIdx.y;
    int index = h + w * blockDim.x * gridDim.x;
    if ((h < rows) && (w < columns))
    {
          result[w] += data[h] * matrix[index];
    }
}
extern "C"
__global__ void derivativeWeight(const float* __restrict__ input, const float* __restrict__ error, float* derWeight, int rows, int columns)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < rows) && (k < columns))
    {
          derWeight[index] += error[j] * input[k];
    }
}
extern "C"
__global__ void findMean_part(const float* __restrict__ A, float* C, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        C[j] += A[index];
    }
}
extern "C"
__global__ void generateErrorNorm(const float* __restrict__ errorNL, const float* __restrict__ gamma, float* errorNorm, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        errorNorm[index] = errorNL[index] * gamma[k];
    }
}
extern "C"
__global__ void derVar_part_2(const float* __restrict__ error, const float* __restrict__ input, const float* __restrict__ mean, float* derVariance, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        derVariance[j] += error[index] * (input[index] - mean[j]);
    }
}
extern "C"
__global__ void derMean_part_2(const float* __restrict__ error, const float* __restrict__ input, const float* __restrict__ mean, float* derMean, float* dVar, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        derMean[j] += error[index];
        dVar[j] += input[index] - mean[j];
    }
}
extern "C"
__global__ void derNorm_part_2(const float* __restrict__ errors, const float* __restrict__ dVar, const float* __restrict__ errorVar, const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ errorMean, float* error, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        error[index] = errors[index] * dVar[j] + errorVar[j] * (input[index] - mean[j]) + errorMean[j];
    }
}
extern "C"
__global__ void derivativeWeight_2(const float* __restrict__ error, const float* __restrict__ output, const float* __restrict__ betta, const float* __restrict__ gamma, float* derBetta, float* derGamma, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        derBetta[k] += error[index];
        derGamma[k] += error[index] * ((output[index] - betta[k]) / gamma[k]);
    }
}
extern "C"
__global__ void findVariance_part(const float* __restrict__ input, const float* __restrict__ mean, float* var, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        float sub = input[index] - mean[j];
        var[j] += sub * sub;
    }
}
extern "C"
__global__ void normalization_part_2(const float* __restrict__ input, const float* __restrict__ mean, const float* __restrict__ varSqrt, float* normOutput, const float* __restrict__ gamma, const float* __restrict__ betta, float* output, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        normOutput[index] = (input[index] - mean[j]) / varSqrt[j];
        output[index] = normOutput[index] * gamma[k] + betta[k];
    }
}
extern "C"
__global__ void add_NNMatrix(const float* __restrict__ matrix, float* data, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    int index = j + k * blockDim.x * gridDim.x;
    if ((j < width) && (k < depth))
    {
        data[k] += matrix[index];
    }
}
extern "C"
__global__ void reverse(float* A, int rows, int columns, int depth)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    int k = blockDim.z * blockIdx.z + threadIdx.z;
    if ((i < (rows / 2 + 1)) && (j < columns) && (k < depth))
    {
       int index = rows - 1 - i;
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
    int k = blockDim.x * blockIdx.x + threadIdx.x;
    if (k < numElements)
    {
       A[k] += B[k];
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
__global__ void vectorScalarSet(float* A, float alpha, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        A[i] = alpha;
    }
}
extern "C"
__global__ void addCopy(const float* __restrict__ A, float* C, int A_col, int C_col, int start, int n) 
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
__global__ void normalization_part_1(float* A, const float* __restrict__ var, float epsilon, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
       A[i] = (float) (sqrt((double)(var[i] + epsilon)));
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
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       float drop = (float) (1.0f / (1.0f - chanceDrop));
       if (random[idx] > chanceDrop)
       {
           A[idx] = A[idx] * drop;
       }
    }
}
extern "C"
__global__ void sub(const float* __restrict__ first, const float* __restrict__ second, float* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       result[idx] = first[idx] - second[idx];
    }
}
extern "C"
__global__ void mul(float* result, float val, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       result[idx] *= val;
    }
}
extern "C"
__global__ void derVar_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       dVar[idx] = (float) (-0.5 * pow((double)(var[idx] + epsilon), -1.5));
    }
}
extern "C"
__global__ void derVar_part_3(const float* __restrict__ dVar, float* derVariance, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       derVariance[idx] *= dVar[idx];
    }
}
extern "C"
__global__ void derMean_part_1(const float* __restrict__ var, float epsilon, float* dMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       dMean[idx] = (float) (-1.0 / sqrt((double)(var[idx] + epsilon)));
    }
}
extern "C"
__global__ void derMean_part_3(const float* __restrict__ dMean, const float* __restrict__ derVar, const float* __restrict__ dVar, int depth, float* derMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       derMean[idx] *= dMean[idx];
       derMean[idx] += (-2.0 * derVar[idx] * dVar[idx]) / (depth);
    }
}
extern "C"
__global__ void derNorm_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       dVar[idx] = (float) (1.0 / sqrt((double)(var[idx] + epsilon)));
    }
}
extern "C"
__global__ void pow2(float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       data[idx] = data[idx] * data[idx];
    }
}
extern "C"
__global__ void sum(const float* __restrict__ data, float* sum, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       sum[0] += data[idx];
    }
}
extern "C"
__global__ void isnan(const float* __restrict__ data, int* result, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       if (result[0] == 0)
       {
           if (isnan(data[idx]))
           {
               result[0] = 1;
           }
       }
    }
}
extern "C"
__global__ void divide_add(float A, const float B, const float C)
{
   A += B / C;
}
extern "C"
__global__ void momentum(float* data, const float* __restrict__ array, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       data[idx] = decay * data[idx] + array[idx] * (1.0 - decay);
    }
}
extern "C"
__global__ void momentumPow2(float* data, const float* __restrict__ vector, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       data[idx] = decay * data[idx] + (1 - decay) * vector[idx] * vector[idx];
    }
}
extern "C"
__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       data[idx] -= (float)((lr / (normN + 0.0000001)) * (nominator[idx]) / (sqrt((double)(denominator[idx] / normD)) + 0.0000001));
    }
}
extern "C"
__global__ void addBackCopy(const float* __restrict__ matrix, int m_column, int column, int start, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       int indexOut = idx * m_column + start * column;
       for (int j = 0; j < column; j++, indexOut++) {
           data[idx] = matrix[indexOut];
       };
    }
}
extern "C"
__global__ void dropoutBack(const float* __restrict__ output, const float* __restrict__ error, float drop, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
       if (output[idx] != 0) {
            data[idx] = error[idx] * drop;
       };
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
__global__ void fillUnderDiagonal(int column, float val, float* data, int numElements)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements)
    {
        for (int j = 0; j < i + 1; j++) {
            data[i * column + j] = val;
        }
    }
}
extern "C"
__global__ void derGelu(const float* __restrict__ input, const float* __restrict__ error, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements)
    {
        float x = input[idx];
        float val = (float) tanh((double)(0.7978846 * x + 0.0356774 * pow((double)x, 3.0)));
        data[idx] = error[idx] * 0.5 * (1.0 + val + x * (1.0 - val * val) * (0.79788846 + 0.1070322 * x * x));
    }
}
extern "C"
__global__ void matrixMultiplicationKernel(const float* __restrict__ A, const float* __restrict__ B, float* C, int width, int P, int Q) {
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if( r < P && c < Q){
        float value = 0;
        for(int k = 0; k < width; k++){
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
       double E = 2.718281828459045;
       for (int i = 0; i < column; i++, index++) {
           data[index] = (float)(pow(E, (double)(input[index] - max)));
               sum += data[index];
       }
       sum += 0.00000001;
       index = k * column;
       for (int i = 0; i < column; i++, index++) {
           data[index] /= sum;
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
