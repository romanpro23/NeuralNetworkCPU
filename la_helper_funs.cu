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
        C[i] = (float) (0.5 * A[i] * (1.0 + tanh((float)(0.7978846 * A[i] + 0.0356774 * pow(A[i], (float)(3.0))))));
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
    const int h = blockDim.x * blockIdx.x + threadIdx.x;
    const int w = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = h * blockDim.y * gridDim.y + w;
    if (index < rows * columns) {
         C[index] += A[w];
    }
}
extern "C"
__global__ void dot_VectorAndMatrix(const float* __restrict__ A, const float* __restrict__ B, float* C, int rows, int columns)
{
    unsigned int h = blockDim.x * blockIdx.x + threadIdx.x;
    if (h < rows) {
       float s = 0;
       int index = h * columns;
       for (int j = 0; j < columns; j++, index++) {
          s += A[j] * B[index];
       }
       C[h] = s;
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
__global__ void findMean_part(const float* __restrict__ A, double* C, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
       double s = 0.0;
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
__global__ void derVar_part_2(const float* __restrict__ error, const float* __restrict__ input, const double* __restrict__ mean, float* derVariance, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        int index = j * depth;
        double m = mean[j];
        float s = 0;
        for (int k = 0; k < depth; k++, index++) {
           s += (float)(error[index] * (input[index] - m));
        }
        derVariance[j] = s;
    }
}
extern "C"
__global__ void derMean_part_2(const float* __restrict__ error, const float* __restrict__ input, const double* __restrict__ mean, float* derMean, double* dVar, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        int index = j * depth;
        float DM = ((float)0);
        double DV = 0.0;
        double m = mean[j];
        for (int k = 0; k < depth; k++, index++) {
           DM += error[index];
           DV += (((double)input[index]) - m);
        }
        derMean[j] = DM;
        dVar[j] = DV;
    }
}
extern "C"
__global__ void derNorm_part_2(const float* __restrict__ errors, const float* __restrict__ dVar, const float* __restrict__ errorVar, const float* __restrict__ input, const double* __restrict__ mean, const float* __restrict__ errorMean, float* error, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int k = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = j * blockDim.y * gridDim.y + k;
    if (index < width * depth) {
        error[index] = errors[index] * dVar[j] + errorVar[j] * ((float)(input[index] - mean[j])) + errorMean[j];
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
__global__ void findVariance_part(const float* __restrict__ input, const double* __restrict__ mean, float* var, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    if (j < width) {
        float s = 0;
        double m = mean[j];
        int index = j * depth;
        for (int k = 0; k < depth; k++, index++) {
           float sub = (float)(input[index] - m);
           s += sub * sub;
        }
        var[j] = s;
    }
}
extern "C"
__global__ void normalization_part_2(const float* __restrict__ input, const double* __restrict__ mean, const float* __restrict__ varSqrt, float* normOutput, const float* __restrict__ gamma, const float* __restrict__ betta, float* output, int width, int depth)
{
    int j = blockDim.x * blockIdx.x + threadIdx.x;
    int k = blockDim.y * blockIdx.y + threadIdx.y;
    if (j < width && k < depth) {
        int index = j * blockDim.y * gridDim.y + k;
        float nO = ((float)(input[index] - mean[j])) / varSqrt[j];
        output[index] = nO * gamma[k] + betta[k];
        normOutput[index] = nO;
    }
}
extern "C"
__global__ void add_NNMatrix(const float* __restrict__ matrix, float* data, int width, int depth)
{
    const int j = blockDim.x * blockIdx.x + threadIdx.x;
    const int k = blockDim.y * blockIdx.y + threadIdx.y;
    const int index = j * blockDim.y * gridDim.y + k;
    if (index < width * depth) {
        data[k] += matrix[index];
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
       float s = A[k] + B[k];
       A[k] = s;
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
__global__ void matrixDiv_doublePrecision(double* A, double B, int numElements)
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
__global__ void normalization_part_1(float* A, const float* __restrict__ var, float epsilon, int numElements)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
       A[i] = (float)sqrt(((float)var[i] + epsilon));
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
       float drop = (float) (1.0 / (1.0 - chanceDrop));
       if (random[idx] > chanceDrop)
       {
           A[idx] = A[idx] * drop;
       }
    }
}
extern "C"
__global__ void sub(const float* __restrict__ first, const float* __restrict__ second, float* result, int numElements)
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
__global__ void derVar_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dVar[idx] = (float) (-0.5 * pow(((double)var[idx] + epsilon), -1.5));
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
       dMean[idx] = (float) (((float)-1.0) / sqrt((float)(var[idx] + epsilon)));
    }
}
extern "C"
__global__ void derMean_part_3(const float* __restrict__ dMean, const float* __restrict__ derVar, const float* __restrict__ dVar, int depth, float* derMean, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       float dM = derMean[idx];
       dM *= dMean[idx];
       dM += (float) ((-2.0 * derVar[idx] * dVar[idx]) / (depth));
       derMean[idx] = dM;
    }
}
extern "C"
__global__ void derNorm_part_1(const float* __restrict__ var, float epsilon, float* dVar, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       dVar[idx] = (float) (1.0 / sqrt((float)(var[idx] + epsilon)));
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
       data[idx] = (float)(decay * data[idx] + array[idx] * (1.0 - decay));
    }
}
extern "C"
__global__ void momentumPow2(float* data, const float* __restrict__ vector, float decay, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] = (float)(decay * data[idx] + (1.0 - decay) * vector[idx] * vector[idx]);
    }
}
extern "C"
__global__ void subDivSqrtNorm(const float* __restrict__ nominator, const float* __restrict__ denominator, float lr, float normN, float normD, float* data, int numElements)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < numElements) {
       data[idx] -= (float)((lr / (normN + 0.0000001)) * (nominator[idx]) / (sqrt((float)(denominator[idx] / normD)) + 0.0000001));
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
        float val = tanh((float)(0.7978846 * x + 0.0356774 * pow((float)x, (float)(3.0))));
        data[idx] = (float)(error[idx] * 0.5 * (1.0 + val + x * (1.0 - val * val) * (0.79788846 + 0.1070322 * x * x)));
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
__global__ void derSoftmax(const float* A, const float* error, float* r, int row, int column)
{
    unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    unsigned int indexI = x * blockDim.y * gridDim.y + y;
    if (x < row && y < column)
    {
        float value = 0.0;
        float AindexI = A[indexI];
        for (int j = 0; j < column; j++) {
           unsigned int indexJ = x * blockDim.y * gridDim.y + j;
           if (y != j) {
               value += error[indexJ] * (AindexI * -A[indexJ]);
           } 
           else {
               value += error[indexJ] * (AindexI * (((float)1.0) - AindexI));
           }
        }
        r[indexI] = value;
   }
}
