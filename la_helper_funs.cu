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
__global__ void softmax_sum(const float* __restrict__ c, int row, int col, float *sum)
{
int X = blockIdx.x*blockDim.x + threadIdx.x;
int Y = blockIdx.y*blockDim.y + threadIdx.y;
if (X < row && Y < col)
{
if(Y == 0)
{
float local_sum = 0;
for(int i = 0; i < col ; i++)
{
local_sum += exp(c[X*col + i]);
}
sum[X] = local_sum;
}
}
}
extern "C"
__global__ void softmax_probability(const float* __restrict__ sum, int row, int col, float* c){
int X = blockIdx.x*blockDim.x + threadIdx.x;
int Y = blockIdx.y*blockDim.y + threadIdx.y;
if (X < row && Y < col){
c[X*col + Y] = exp(c[X*col + Y]) / sum[X];
}
}
extern "C"
__global__ void softmax_kernel(const float* __restrict__ input_data, int nrow, int ncol, float* output_data)
{
int y = blockIdx.x * blockDim.x * blockDim.y + threadIdx.y * blockDim.x + threadIdx.x;
if (y >= nrow) {
return;
}
input_data += y * ncol;
output_data += y * ncol;
float maxval = *input_data;
for (int x = 1; x < ncol; ++x) {
maxval = max(maxval, input_data[x]);
}
float sum = 0;
for (int x = 0; x < ncol; ++x) {
sum += exp(input_data[x] - maxval);
}
for (int x = 0; x < ncol; ++x) {
output_data[x] = exp(input_data[x] - maxval) / sum;
}
}
extern "C"
__global__ void set_value(float* A, float value, long int total_ops)
{
int thread_id_x = threadIdx.x +blockIdx.x*blockDim.x;
if (thread_id_x < total_ops)
A[thread_id_x]=value;
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
__global__ void dot(const float* __restrict__ a, const float* __restrict__ b, float* c, int a_ncolumns, int c_nlines, int c_ncolumns, int nBlocks)
{
    int i, z; 
    float sum = 0;
    const int NTHREADS_X = 32;
    const int NTHREADS_Y = 32;
    int nMultiplications = a_ncolumns;
    int multiplicationsInBlock = NTHREADS_Y;
    int column = blockIdx.x * blockDim.x + threadIdx.x;
    int line =  blockIdx.y * blockDim.y + threadIdx.y;
    __shared__ float s_a[NTHREADS_Y][NTHREADS_X];
    __shared__ float s_b[NTHREADS_Y][NTHREADS_X];
    int a_tLine, a_tColumn, b_tLine, b_tColumn;
    for (z = 0; z < nBlocks; z++)
    {
        a_tLine = (blockIdx.y * NTHREADS_Y + threadIdx.y);
        a_tColumn = (z * NTHREADS_X + threadIdx.x);
        if (a_tLine < c_nlines && a_tColumn < a_ncolumns)
        {
            s_a[threadIdx.y][threadIdx.x] = a[ (a_ncolumns * a_tLine) + a_tColumn];
        }
        b_tLine = (z * NTHREADS_Y + threadIdx.y);
        b_tColumn = (blockIdx.x * NTHREADS_X + threadIdx.x);
        if (b_tLine < a_ncolumns && b_tColumn < c_ncolumns)
        {
            s_b[threadIdx.y][threadIdx.x] = b[ (c_ncolumns * b_tLine) + b_tColumn ];
        }
        __syncthreads();
        if (column < c_ncolumns && line < c_nlines)
        {
            if (nMultiplications < NTHREADS_Y)
            {
                multiplicationsInBlock = nMultiplications;
            }
            for (i = 0; i < multiplicationsInBlock; i++)
            {
                sum += s_a[threadIdx.y][i] * s_b[i][threadIdx.x];
            }
            nMultiplications -= NTHREADS_Y;
        }
        __syncthreads();
    }
    if (column < c_ncolumns && line < c_nlines)
    {
        c[line * c_ncolumns + column] = sum;
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
__global__ void matrixMultiplyShared(float * A, float * B, float * C, int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns) {
const int TILE_WIDTH = 32;
__shared__ float sA[TILE_WIDTH][TILE_WIDTH];
__shared__ float sB[TILE_WIDTH][TILE_WIDTH];
int Row = blockDim.y * blockIdx.y + threadIdx.y;
int Col = blockDim.x * blockIdx.x + threadIdx.x;
float Cvalue = 0.0;
sA[threadIdx.y][threadIdx.x] = 0.0;
sB[threadIdx.y][threadIdx.x] = 0.0;
for (int ph = 0; ph < (((numAColumns - 1) / TILE_WIDTH) + 1); ph++) {
    if ((Row < numARows) && (threadIdx.x + (ph * TILE_WIDTH)) < numAColumns) {
        sA[threadIdx.y][threadIdx.x] = A[(Row * numAColumns) + threadIdx.x + (ph * TILE_WIDTH)];
    } else {
        sA[threadIdx.y][threadIdx.x] = 0.0;
    }
    if (Col < numBColumns && (threadIdx.y + ph * TILE_WIDTH) < numBRows) {
        sB[threadIdx.y][threadIdx.x] = B[(threadIdx.y + ph * TILE_WIDTH) * numBColumns + Col];
    } else {
        sB[threadIdx.y][threadIdx.x] = 0.0;
    }
    __syncthreads();
    for (int j = 0; j < TILE_WIDTH; ++j) {
        Cvalue += sA[threadIdx.y][j] * sB[j][threadIdx.x];
    }
}
if (Row < numCRows && Col < numCColumns) {
    C[Row * numCColumns + Col] = Cvalue;
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

