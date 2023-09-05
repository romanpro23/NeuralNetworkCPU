package nnarrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
import jcuda.jcublas.cublasOperation;
import jcuda.runtime.JCuda;
import lombok.Getter;
import lombok.SneakyThrows;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Scanner;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static utilities.GPUInit.*;
import static utilities.GPUInit.allocatedUse;

public class NNMatrix extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    public int[] rowIndex;

    public NNMatrix(int row, int column) {
        super(column * row);
        this.column = column;
        this.row = row;

        if (!Use.GPU) {
            rowIndex = new int[row];
            for (int i = 0; i < row; i++) {
                rowIndex[i] = i * column;
            }
        }
        else
        {
            allocatedPut();
        }

        countAxes = 2;
    }

    public NNMatrix(int row, int column, float[] data) {
        super(data);
        this.column = column;
        this.row = row;

        if (!Use.GPU) {
            rowIndex = new int[row];
            for (int i = 0; i < row; i++) {
                rowIndex[i] = i * column;
            }
        }
        else
        {
            allocatedPut();
        }

        countAxes = 2;
    }

    public void allocatedPut() {
        Use U = new Use();
        U.data_gpu = this.data_gpu;
        U.HashCode = this.hashCode();
        allocated.put(String.valueOf(this.hashCode()), new WeakReference<>(this));
        allocatedUse.put(String.valueOf(this.hashCode()), U);
    }

    public NNMatrix(NNMatrix matrix) {
        this(matrix.row, matrix.column);
    }

    public void fillUnderDiagonal(float val) {
        if (!Use.GPU) {
            for (int i = 0; i < row; i++) {
                for (int j = 0; j < i + 1; j++) {
                    set(i, j, val);
                }
            }
        }
        else
        {
            int n = row;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "fillUnderDiagonal");
            Pointer kernelParameters = Pointer.to(Pointer.to(new int[]{column}), Pointer.to(new float[]{val}), Pointer.to(this.data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }
    }

    public void mask(NNMatrix mask, float val, float newVal) {
        if (!Use.GPU) {
            for (int i = 0; i < size; i++) {
                if (mask.data[i] == val) {
                    data[i] = newVal;
                }
            }
        }
        else
        {
            int n = size;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "mask");
            Pointer kernelParameters = Pointer.to(Pointer.to(mask.data_gpu), Pointer.to(new float[]{val}), Pointer.to(new float[]{newVal}), Pointer.to(this.data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
        }
    }

    public NNMatrix mask(NNVector mask) {
        NNMatrix result = new NNMatrix(row, column);
        for (int i = 0, index = 0; i < row; i++) {
            for (int j = 0; j < column; j++, index++) {
                result.data[index] = data[index] * mask.get(i);
            }
        }
        return result;
    }

    public NNVector[] toVectors() {
        NNVector[] vectors = new NNVector[row];
        for (int i = 0; i < row; i++) {
            vectors[i] = new NNVector(column);
            System.arraycopy(data, rowIndex[i], vectors[i].data, 0, column);
        }

        return vectors;
    }

    public void copy(NNMatrix matrix, int start) {
        System.arraycopy(matrix.data, 0, data, start, matrix.size);
    }

    public void addCopy(NNMatrix matrix, int start) {
        if (!Use.GPU) {
            int indexIn, indexOut = 0;
            for (int i = 0; i < row; i++) {
                indexIn = rowIndex[i] + start * matrix.column;
                for (int j = 0; j < matrix.column; j++, indexIn++, indexOut++) {
                    data[indexIn] = matrix.data[indexOut];
                }
            }
        }
        else
        {
            int n = row;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "addCopy");
            Pointer kernelParameters = Pointer.to(Pointer.to(matrix.data_gpu), Pointer.to(this.data_gpu), Pointer.to(new int[]{matrix.column}), Pointer.to(new int[]{this.column}), Pointer.to(new int[]{start}), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void addBackCopy(NNMatrix matrix, int start) {
        if (!Use.GPU) {
            int indexIn = 0, indexOut;
            for (int i = 0; i < row; i++) {
                indexOut = matrix.rowIndex[i] + start * column;
                for (int j = 0; j < column; j++, indexIn++, indexOut++) {
                    data[indexIn] = matrix.data[indexOut];
                }
            }
        }
        else
        {
            int n = row;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "addBackCopy");
            Pointer kernelParameters = Pointer.to(Pointer.to(matrix.data_gpu), Pointer.to(new int[]{matrix.column}), Pointer.to(new int[]{this.column}), Pointer.to(new int[]{start}), Pointer.to(this.data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void set(NNVector vector, int index_t) {
        int index = rowIndex[index_t];
        System.arraycopy(vector.data, 0, data, index, vector.size);
    }

    public void set(NNMatrix matrix, int index_t) {
        int index = rowIndex[index_t];
        System.arraycopy(matrix.data, 0, data, index, matrix.size);
    }

    @Override
    public int[] shape() {
        return new int[]{row, column};
    }

    public float get(int i, int j) {
        return data[rowIndex[i] + j];
    }

    public void set(int i, int j, float val) {
        data[rowIndex[i] + j] = val;
    }

    @SneakyThrows
    public void add(NNMatrix matrix) {
        if (size != matrix.size) {
            throw new Exception("Vector has difference size");
        }

        if (!Use.GPU) {
            for (int i = 0; i < size; i++) {
                data[i] += matrix.data[i];
            }
        }
        else
        {
            MatAdd(matrix);
        }
    }

    public void addScalarMul(NNTensor input, NNMatrix matrix) {
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                float scalarDot = 0;
                for (int k = 0; k < input.getDepth(); k++) {
                    scalarDot += input.get(i, j, k) * matrix.get(i, k);
                }
                add(i, j, scalarDot);
            }
        }
    }

    public NNMatrix derScalarMul(NNTensor input, NNMatrix error) {
        NNMatrix result = new NNMatrix(row, column);
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                for (int k = 0; k < input.getDepth(); k++) {
                    result.add(i, k, input.get(i, j, k) * error.get(i, j));
                }
            }
        }

        return result;
    }

    public void squash(NNMatrix matrix) {
        for (int i = 0; i < row; i++) {
            float mod = 0;
            for (int j = 0; j < column; j++) {
                mod += Math.pow(matrix.get(i, j), 2);
            }
            float scale = (float) ((mod / (1f + mod)) / Math.sqrt(mod + 0.0000001f));

            for (int j = 0; j < column; j++) {
                set(i, j, scale * matrix.get(i, j));
            }
        }
    }

    public void derSquash(NNMatrix matrix, NNMatrix error) {
        for (int i = 0; i < row; i++) {
            float mod_2 = 0;
            for (int j = 0; j < column; j++) {
                mod_2 += Math.pow(matrix.get(i, j), 2);
            }
            float mod = (float) Math.sqrt(mod_2) + 0.00000001f;
            float _mod_2 = 2 * mod;
            float mod_2_one = mod_2 + 1;
            float mod_2_one_2 = mod_2_one * mod_2_one;

            for (int j = 0; j < column; j++) {
                float x = matrix.get(i, j);
                float x_2 = x * x;
                set(i, j, ((mod_2_one * (mod + x_2 / mod) - x_2 * _mod_2) / (mod_2_one_2)) * error.get(i, j));
            }
        }
    }

    public NNVector mod() {
        NNVector result = new NNVector(row);
        for (int i = 0; i < row; i++) {
            float mod = 0;
            for (int j = 0; j < column; j++) {
                mod += Math.pow(get(i, j), 2);
            }
            result.set(i, (float) Math.sqrt(mod) + 0.00000001f);
        }
        return result;
    }

    public NNMatrix backMod(NNVector output, NNVector error) {
        NNMatrix result = new NNMatrix(row, column);
        for (int i = 0; i < row; i++) {
            float err = error.get(i) / (output.get(i) + 0.00000001f);
            for (int j = 0; j < column; j++) {
                result.set(i, j, get(i, j) * err);
            }
        }
        return result;
    }

    public void add(int i, int j, float val) {
        data[rowIndex[i] + j] += val;
    }

    public NNMatrix transpose() {
        NNMatrix nnMatrix = new NNMatrix(this.column, this.row);

        if (!Use.GPU) {
            int index;
            for (int i = 0; i < row; i++) {
                index = rowIndex[i];
                for (int j = 0; j < column; j++, index++) {
                    nnMatrix.data[i + nnMatrix.rowIndex[j]] = data[index];
                }
            }
        }
        else
        {
            JCublas2.cublasSgeam(cublasHandle, cublasOperation.CUBLAS_OP_T, cublasOperation.CUBLAS_OP_T, column, row, Pointer.to(new float[]{1.0f}), data_gpu, row, Pointer.to(new float[]{0.0f}), new Pointer(), row, nnMatrix.data_gpu, nnMatrix.row);
            IsNan();
            IsNan(nnMatrix);
        }

        return nnMatrix;
    }

    public void save(FileWriter writer) throws IOException {
        writer.write(row + " " + column + "\n");
        for (int i = 0; i < row; i++) {
            for (int j = 0; j < column; j++) {
                writer.write(data[rowIndex[i] + j] + " ");
                if (j % 1000 == 0) {
                    writer.flush();
                }
            }
            writer.write("\n");
            writer.flush();
        }
    }

    public NNMatrix dot_2(NNMatrix matrix)
    {
        NNMatrix result = new NNMatrix(row, matrix.column);

        if (column == matrix.row) {
            for (int i = 0; i < row; ++i) {
                for (int j = 0; j < result.column; ++j) {
                    float tmp = 0.0f;
                    for (int h = 0; h < column; ++h) {
                        tmp += data[i * matrix.row + h] * matrix.data[h * result.column + j];
                    }
                    result.data[i * result.column + j] = tmp;
                }
            }
        }
        return result;
    }

    public NNMatrix dotT(NNMatrix matrix) {
        NNMatrix result = new NNMatrix(row, matrix.getRow());

        if (!Use.GPU) {
            for (int n = 0, indR = 0; n < row; n++) {
                for (int i = 0, index = 0; i < matrix.getRow(); i++, indR++) {
                    for (int j = 0, indI = rowIndex[n]; j < matrix.getColumn(); j++, index++, indI++) {
                        result.data[indR] += data[indI] * matrix.data[index];
                    }
                }
            }
        }
        else {
            NNMatrix transpose = matrix.transpose();
            gemm(1.0f, this, transpose, 0.0f, result);

            IsNan(result);
        }

        return result;
    }

    public NNMatrix dot(NNTensor tensor) {
        NNMatrix result = new NNMatrix(tensor.getRows(), tensor.getDepth());

        for (int i = 0; i < tensor.getRows(); i++) {
            for (int l = 0; l < tensor.getDepth(); l++) {
                for (int j = 0; j < tensor.getColumns(); j++) {
                    result.add(i, l, get(i, j) * tensor.get(i, j, l));
                }
            }
        }
        return result;
    }

    public NNMatrix dotT(NNTensor tensor) {
        NNMatrix result = new NNMatrix(tensor.getRows(), tensor.getColumns());

        for (int i = 0; i < tensor.getRows(); i++) {
            for (int j = 0; j < tensor.getColumns(); j++) {
                for (int l = 0; l < tensor.getDepth(); l++) {
                    result.add(i, j, get(i, l) * tensor.get(i, j, l));
                }
            }
        }
        return result;
    }

    public NNTensor dotR(NNMatrix matrix) {
        NNTensor result = new NNTensor(row, column, matrix.column);

        for (int i = 0; i < row; i++) {
            for (int l = 0; l < matrix.column; l++) {
                for (int j = 0; j < column; j++) {
                    result.add(i, j, l, get(i, j) * matrix.get(i, l));
                }
            }
        }
        return result;
    }

    public NNTensor dot(NNTensor4D weight) {
        NNTensor result = new NNTensor(weight.depth(), weight.length(), weight.row());

        for (int i = 0; i < weight.depth(); i++) {
            for (int j = 0; j < weight.length(); j++) {
                for (int l = 0; l < weight.row(); l++) {
                    for (int k = 0; k < weight.column(); k++) {
                        result.add(i, j, l, get(j, k) * weight.get(i, j, l, k));
                    }
                }
            }
        }
        return result;
    }

    public NNMatrix derCapsuleAffineTransform(NNTensor4D weight, NNTensor error) {
        NNMatrix result = new NNMatrix(row, column);

        for (int i = 0; i < weight.depth(); i++) {
            for (int j = 0; j < weight.length(); j++) {
                for (int k = 0; k < weight.column(); k++) {
                    for (int l = 0; l < weight.row(); l++) {
                        result.add(j, k, error.get(i, j, l) * weight.get(i, j, l, k));
                    }
                }
            }
        }
        return result;
    }

    public NNMatrix dot(NNMatrix matrix) {
        if (!Use.GPU) {
            return dotT(matrix.transpose());
        }
        else {
            NNMatrix result = new NNMatrix(row, matrix.column);
            gemm(1.0f, this, matrix, 0.0f, result);

            IsNan(result);
            return result;
        }
    }

    // C = alpha * A * B + beta * C
    private static void gemm(float alpha, NNMatrix A, NNMatrix B, float beta, NNMatrix C) {

        //NNMatrix D = new NNMatrix(C.row, C.column);
        JCublas2.cublasSgemm(cublasHandle, cublasOperation.CUBLAS_OP_N, cublasOperation.CUBLAS_OP_N, B.column, A.row, B.row, Pointer.to(new float[]{alpha}), B.data_gpu, B.column, A.data_gpu, B.row, Pointer.to(new float[]{beta}), C.data_gpu, B.column);

        /*CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "matrixMultiplicationKernel");
        Pointer kernelParameters = Pointer.to(Pointer.to(A.data_gpu), Pointer.to(B.data_gpu), Pointer.to(C.data_gpu), Pointer.to(new int[]{A.column}), Pointer.to(new int[]{C.row}), Pointer.to(new int[]{C.column}));
        int blockSizeX = (int) Math.min(C.row, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int blockSizeY = (int) Math.min(C.column, Math.pow(BLOCK_SIZE, (double) 1 / 2));
        int gridSizeX = (int) Math.ceil((double) C.row / blockSizeX);
        int gridSizeY = (int) Math.ceil((double) C.column / blockSizeY);

        cuLaunchKernel(function,
                gridSizeX, gridSizeY, 1,      // Grid dimension
                blockSizeX, blockSizeY, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );*/
        //if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
    }

    public NNMatrix dot(NNVector vector) {
        NNMatrix result = new NNMatrix(row, vector.size);
        float val;

        for (int n = 0, indR = 0; n < row; n++) {
            for (int i = 0; i < vector.size; i++, indR++) {
                val = 0;
                for (int j = 0; j < column; j++) {
                    val += get(n, j) * vector.get(i);
                }
                result.data[indR] = val;
            }
        }
        return result;
    }

    public NNMatrix dotT(NNVector vector) {
        NNMatrix result = new NNMatrix(row, 1);
        float val;

        for (int n = 0; n < row; n++) {
            val = 0;
            for (int i = 0; i < column; i++) {
                val += get(n, i) * vector.get(i);
            }
            result.data[n] = val;
        }
        return result;
    }

    public static NNMatrix read(Scanner scanner) {
        int[] size = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        NNMatrix matrix = new NNMatrix(size[0], size[1]);
        for (int i = 0; i < matrix.row; i++) {
            double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
            for (int j = 0; j < matrix.column; j++) {
                matrix.data[matrix.rowIndex[i] + j] = (float) arr[j];
            }
        }
        return matrix;
    }

    public void convolution(NNMatrix input, NNTensor weight, int step, int pad) {
        int x0, inputIndex, weightIndex, outputIndex;
        float val;

        for (int x = -pad, w = 0; w < row; x += step, w++) {
            outputIndex = rowIndex[w];
            for (int d = 0; d < weight.getRows(); d++, outputIndex++) {
                val = 0;
                for (int j = 0; j < weight.getColumns(); j++) {
                    x0 = x + j;
                    if (x0 < 0 || x0 >= input.row) {
                        continue;
                    }
                    weightIndex = weight.getRowsIndex()[d] + weight.getColumnsIndex()[j];
                    inputIndex = input.rowIndex[x0];
                    for (int c = 0; c < weight.getDepth(); c++, inputIndex++, weightIndex++) {
                        val += input.data[inputIndex] * weight.data[weightIndex];
                    }
                }
                data[outputIndex] = val;
            }
        }
    }

    public void transposeConvolution(NNMatrix input, NNTensor weight, int padding) {
        int x0, inputIndex, weightIndex, outputIndex;
        int pad = weight.getColumns() - 1 - padding;
        int sCore = weight.getColumns() - 1;
        int sC;

        float val;

        for (int x = -pad, w = 0; w < row; x++, w++) {
            outputIndex = rowIndex[w];
            for (int d = 0; d < weight.getDepth(); d++, outputIndex++) {
                val = 0;
                for (int j = 0; j < weight.getColumns(); j++) {
                    x0 = x + j;
                    if (x0 < 0 || x0 >= input.row) {
                        continue;
                    }
                    sC = sCore - j;
                    weightIndex = weight.getColumnsIndex()[sC] + d;
                    inputIndex = input.rowIndex[x0];

                    for (int c = 0; c < weight.getRows(); c++, inputIndex++) {
                        val += input.data[inputIndex] * weight.data[weight.getRowsIndex()[c] + weightIndex];
                    }
                }
                data[outputIndex] = val;
            }
        }
    }

    @SneakyThrows
    public void add(NNVector vector) {
        if (column != vector.size) {
            throw new Exception("Array has difference size");
        }
        if (!Use.GPU) {
            int inputIndex = 0;
            for (int i = 0; i < row; i++) {
                for (int k = 0; k < column; k++, inputIndex++) {
                    data[inputIndex] += vector.data[k];
                }
            }
        }
        else
        {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "add3");
            Pointer kernelParameters = Pointer.to(Pointer.to(vector.data_gpu), Pointer.to(data_gpu),  Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            int blockSizeX = (int) Math.min(row, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            int blockSizeY = (int) Math.min(column, Math.pow(BLOCK_SIZE, (double) 1 / 2));
            int gridSizeX = (int) Math.ceil((double) row / blockSizeX);
            int gridSizeY = (int) Math.ceil((double) column / blockSizeY);

            cuLaunchKernel(function,
                    gridSizeX, gridSizeY, 1,      // Grid dimension
                    blockSizeX, blockSizeY, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    @SneakyThrows
    public NNMatrix sum(NNVector vector) {
        if (column != vector.size) {
            throw new Exception("Array has difference size");
        }
        NNMatrix result = new NNMatrix(this);
        int inputIndex = 0;
        for (int i = 0; i < row; i++) {
            for (int k = 0; k < column; k++, inputIndex++) {
                result.data[inputIndex] = data[inputIndex] + vector.data[k];
            }
        }

        return result;
    }

    @SneakyThrows
    public NNVector sum() {
        NNVector result = new NNVector(column);
        int inputIndex = 0;
        for (int i = 0; i < row; i++) {
            for (int k = 0; k < column; k++, inputIndex++) {
                result.data[k] = data[inputIndex];
            }
        }

        return result;
    }

    public void backGlobalMaxPool(NNMatrix input, NNVector output, NNVector error) {
        int index = 0;
        for (int i = 0; i < input.row; i++) {
            for (int k = 0; k < input.column; k++, index++) {
                if (output.data[k] == input.data[index]) {
                    data[index] = error.data[k];
                }
            }
        }
    }

    public void backGlobalAveragePool(NNVector error) {
        int index = 0;
        error.div(row);
        for (int i = 0; i < row; i++) {
            for (int k = 0; k < column; k++, index++) {
                data[index] = error.data[k];
            }
        }
    }

    public void addMulT(int n_row, NNVector vector, NNMatrix matrix) {
        for (int i = 0, index = 0; i < matrix.getRow(); i++) {
            for (int j = 0, indexOutput = rowIndex[n_row]; j < matrix.getColumn(); j++, index++, indexOutput++) {
                data[indexOutput] += vector.data[i] * matrix.data[index];
            }
        }
    }

    public NNMatrix stride(int stride) {
        if (stride == 1) {
            return this;
        }
        NNMatrix result = new NNMatrix(row * stride, column);
        int inputIndex, outpuIndex;
        for (int i = 0; i < row; i++) {
            inputIndex = rowIndex[i];
            outpuIndex = result.rowIndex[i * stride];
            for (int k = 0; k < column; k++, inputIndex++, outpuIndex++) {
                result.data[outpuIndex] = data[inputIndex];
            }
        }
        return result;
    }

    public void softmax(NNMatrix input) {
        if (!Use.GPU) {
            int index;
            for (int k = 0; k < row; k++) {
                float sum = 0;
                index = k * column;
                float max = input.data[index];
                for (int i = 1; i < column; i++, index++) {
                    if (max < input.data[index])
                        max = input.data[index];
                }
                index = k * column;
                for (int i = 0; i < column; i++, index++) {
                    double val = (Math.pow(Math.E, input.data[index] - max));
                    if (val > Float.MAX_VALUE) {
                        data[index] = Float.MAX_VALUE;
                    } else {
                        data[index] = (float) val;
                    }

                    if (sum + data[index] > Float.MAX_VALUE) {
                        sum = Float.MAX_VALUE;
                    } else {
                        sum += data[index];
                    }
                }
                sum += 0.00000001f;

                index = k * column;
                for (int i = 0; i < column; i++, index++) {
                    data[index] /= sum;
                }
            }
        }
        else
        {;
            int n = row;
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "Softmax");
            Pointer kernelParameters = Pointer.to(Pointer.to(input.data_gpu), Pointer.to(this.data_gpu), Pointer.to(new int[]{column}),  Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

            IsNan();
        }
    }

    public void derSoftmax(NNMatrix output, NNMatrix error) {
        if (!Use.GPU) {
            int index, indexI, indexJ;
            for (int k = 0; k < row; k++) {
                float value;
                index = k * column;
                indexI = index;
                for (int i = 0; i < column; i++, indexI++) {
                    data[indexI] = 0;
                    indexJ = index;
                    for (int j = 0; j < column; j++, indexJ++) {
                        if (i != j) {
                            value = output.data[indexI] * -output.data[indexJ];
                        } else {
                            value = output.data[indexI] * (1 - output.data[indexI]);
                        }
                        data[indexI] += error.getData()[indexJ] * value;
                    }
                }
            }
        }
        else
        {
            derSoftmax(output, error, this);
        }
    }

    private static void derSoftmax(NNMatrix A, NNMatrix auxE, NNMatrix C) {
        int n = A.row;
        CUfunction function = new CUfunction();
        cuModuleGetFunction(function, helperModule, "derSoftmax");
        Pointer kernelParameters = Pointer.to(Pointer.to(A.data_gpu), Pointer.to(auxE.data_gpu), Pointer.to(new int[]{A.column}), Pointer.to(C.data_gpu), Pointer.to(new int[]{n}));
        int blockSize = Math.min(n, BLOCK_SIZE);
        int gridSizeX = (int) Math.ceil((double) n / blockSize);

        cuLaunchKernel(function,
                gridSizeX, 1, 1,      // Grid dimension
                blockSize, 1, 1,      // Block dimension
                0, null,               // Shared memory size and stream
                kernelParameters, null // Kernel- and extra parameters
        );
        if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();

        A.IsNan(C);
    }

    @Override
    public String toString() {
        return "NNMatrix [" +
                "size: (" + row +
                ", " + column +
                "), data: " + Arrays.toString(data) +
                ']';
    }

    public float[] toArray() {
        float[] data_h = new float[row * column];
        JCublas2.cublasGetVector(data_h.length, Sizeof.FLOAT, data_gpu, 1, Pointer.to(data_h), 1);
        return data_h;
    }

    public float[][] toArray2() {
        float[] data_h = toArray();
        return fromColMajor(data_h, row);
    }

    private static float[][] fromColMajor(float[] data, int rows) {
        int cols = data.length / rows;
        float[][] mat = new float[rows][cols];
        int i = 0;
        for (int c = 0; c < cols; ++c) {
            for (int r = 0; r < rows; ++r) {
                mat[r][c] = data[i];
                i++;
            }
        }
        return mat;
    }

    public void free() {
        //if (data_gpu != null) JCuda.cudaFree(data_gpu);
        //if (rowsIndex_gpu != null) JCuda.cudaFree(rowsIndex_gpu);
    }
}
