package nnarrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.cudaDataType;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.*;
import jcuda.runtime.JCuda;
import jcuda.runtime.dim3;
import lombok.SneakyThrows;
import utilities.JCudaHelper;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

import static jcuda.driver.JCudaDriver.cuLaunchKernel;
import static jcuda.driver.JCudaDriver.cuModuleGetFunction;
import static jcuda.runtime.JCuda.*;
import static jcuda.runtime.cudaFuncAttribute.cudaFuncAttributeMaxDynamicSharedMemorySize;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static utilities.GPUInit.cublasHandle;
import static utilities.GPUInit.helperModule;

public class NNVector extends NNArray {

    public NNVector(int length) {
        super(length);
        countAxes = 1;
    }

    public NNVector(int length, boolean TYPE) {
        super(length, TYPE);
        countAxes = 1;
    }

    public NNVector(NNVector vector) {
        super(vector.size);
        countAxes = 1;
    }

    public NNVector(NNVector vector, boolean TYPE) {
        super(vector.size, TYPE);
        countAxes = 1;
    }

    public NNVector(NNArray array) {
        super(array.data, array.sdata);
        countAxes = 1;
    }

    public NNVector(NNArray array, boolean TYPE) {
        super(array.data, array.sdata, TYPE);
        countAxes = 1;
    }

    public NNVector(float[] _data, short[] _sdata) {
        super(_data, _sdata);
        countAxes = 1;
    }

    public NNVector(float[] _data, short[] _sdata, boolean TYPE) {
        super(_data, _sdata, TYPE);
        countAxes = 1;
    }

    public NNVector(int[] _data) {
        super(_data);
        countAxes = 1;
    }

    public NNVector dot(NNMatrix matrix) {
        NNVector result = new NNVector(matrix.getRow(), matrix.TYPE);

        if (Use.CPU) {
            for (int i = 0, index = 0; i < matrix.getRow(); i++) {
                for (int j = 0; j < matrix.getColumn(); j++, index++) {
                    result.data[i] += data[j] * matrix.data[index];
                }
            }
        }
        if (Use.GPU) {
            if (!matrix.TYPE) {
                int row =  matrix.getRow();
                int column =  matrix.getColumn();
                CUfunction function = new CUfunction();
                cuModuleGetFunction(function, helperModule, "dot_VectorAndMatrix");
                Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(matrix.data_gpu), Pointer.to(result.data_gpu),  Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
                int blockSizeX = (int) Math.min(row, Math.pow(BLOCK_SIZE, 1));
                int gridSizeX = (int) Math.ceil((double) row / blockSizeX);

                cuLaunchKernel(function,
                        gridSizeX, 1, 1,      // Grid dimension
                        blockSizeX, 1, 1,      // Block dimension
                        0, null,               // Shared memory size and stream
                        kernelParameters, null // Kernel- and extra parameters
                );
                if (Use.DEBUG_SYNC) JCudaDriver.cuCtxSynchronize();
            }
        }

        return result;
    }


    public void addMulRowToMatrix(NNMatrix input, int n_row, NNMatrix matrix) {
        for (int i = 0, indexMatrix = 0; i < size; i++) {
            for (int j = 0, indexInput = input.getRowIndex()[n_row]; j < matrix.getColumn(); j++, indexInput++, indexMatrix++) {
                data[i] += input.data[indexInput] * matrix.data[indexMatrix];
            }
        }
    }

    @SneakyThrows
    public void addRowFromMatrix(NNMatrix input, int n_row) {
        if (size != input.getColumn()) {
            throw new Exception("Vector and Matrix has difference size");
        }
        for (int j = 0, indexInput = input.getRowIndex()[n_row]; j < size; j++, indexInput++) {
            data[j] += input.data[indexInput];
        }
    }

    public NNVector concat(NNVector vector){
        NNVector result = new NNVector(size + vector.size());
        System.arraycopy(data, 0, result.data, 0, size);
        System.arraycopy(vector.data, 0, result.data, size, vector.size);

        return result;
    }

    public NNVector subVector(int startPos, int size){
        NNVector result = new NNVector(size);
        System.arraycopy(data, startPos, result.data, 0, size);

        return result;
    }

    public void ClearCpuData()
    {
        data = null;
        sdata = null;
    }

    @SneakyThrows
    public void setRowFromMatrix(NNMatrix input, int n_row) {
        if (size != input.getColumn()) {
            throw new Exception("Vector and Matrix has difference size");
        }
        System.arraycopy(input.data, input.getRowIndex()[n_row], data, 0, size);
    }

    public void addMul(NNVector input, NNMatrix matrix) {
        for (int i = 0, indexMatrix = 0; i < size; i++) {
            for (int j = 0; j < matrix.getColumn(); j++, indexMatrix++) {
                data[i] += input.data[j] * matrix.data[indexMatrix];
            }
        }
    }

    @SneakyThrows
    public void mulVectors(NNVector first, NNVector second) {
        if (size != first.size || size != second.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] = first.data[i] * second.data[i];
        }
    }

    @SneakyThrows
    public void mulNegativeVectors(NNVector first, NNVector second) {
        if (size != first.size || size != second.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] = -first.data[i] * second.data[i];
        }
    }

    @SneakyThrows
    public void addMulUpdateVectors(NNVector updateVector, NNVector second) {
        if (size != updateVector.size || size != second.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] += (1 - updateVector.data[i]) * second.data[i];
        }
    }

    @SneakyThrows
    public void setMulUpdateVectors(NNVector updateVector, NNVector second) {
        if (size != updateVector.size || size != second.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] = (1 - updateVector.data[i]) * second.data[i];
        }
    }

    @SneakyThrows
    public void mulUpdateVector(NNVector updateVector) {
        if (size != updateVector.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] *= (1 - updateVector.data[i]);
        }
    }

    public void momentumAverage(NNArray array, final float decay) {
        for (int i = 0; i < size; i++) {
            data[i] += (array.data[i] - data[i]) * decay;
        }
    }

    public NNVector dotT(NNMatrix matrix) {
        NNVector result = new NNVector(matrix.getColumn());

        if (Use.CPU) {
            for (int i = 0, index = 0; i < matrix.getRow(); i++) {
                for (int j = 0; j < matrix.getColumn(); j++, index++) {
                    result.data[j] += data[i] * matrix.data[index];
                }
            }
        }
        if (Use.GPU) {
            int row = matrix.getRow();
            int column = matrix.getColumn();
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "dotT_VectorAndMatrix");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(matrix.data_gpu), Pointer.to(result.data_gpu),  Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            int blockSizeX = (int) Math.min(column, Math.pow(BLOCK_SIZE, 1));
            int gridSizeX = (int) Math.ceil((double) column / blockSizeX);

            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan_float();
                IsNan_float(matrix);
                IsNan_float(result);
            }
            //System.out.println(" ! " + (System.nanoTime() - start0) / 1000 + " ! " + 0);
        }

        return result;
    }

    public float mod(){
        float sum = 0;
        for (int i = 0; i < size; i++) {
            sum += get(i);
        }
        return (float) Math.sqrt(sum);
    }

    public NNVector squash(){
        NNVector result = new NNVector(size);
        float mod = mod();
        float scale = (mod * mod / (1f + mod * mod)) / (mod + 0.00000001f);
        for (int i = 0; i < size; i++) {
            result.set(i, get(i) * scale);
        }

        return result;
    }

    public void addMulT(NNVector vector, NNMatrix matrix) {
        for (int i = 0, index = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++, index++) {
                data[j] += vector.data[i] * matrix.data[index];
            }
        }
    }

    public void globalMaxPool(NNTensor input) {
        int index = 0;
        fill(-1000);
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getColumns(); j++) {
                for (int k = 0; k < size; k++, index++) {
                    if (data[k] < input.data[index]) {
                        data[k] = input.data[index];
                    }
                }
            }
        }
    }

    public void globalMaxPool(NNMatrix input) {
        int index = 0;
        fill(-1000);
        for (int i = 0; i < input.getRow(); i++) {
            for (int k = 0; k < size; k++, index++) {
                if (data[k] < input.data[index]) {
                    data[k] = input.data[index];
                }
            }
        }
    }

    public void globalAveragePool(NNTensor input) {
        int index = 0;
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getColumns(); j++) {
                for (int k = 0; k < size; k++, index++) {
                    data[k] += input.data[index];
                }
            }
        }
        div(input.getRows() * input.getColumns());
    }

    public void globalAveragePool(NNMatrix input) {
        int index = 0;
        for (int i = 0; i < input.getRow(); i++) {
            for (int k = 0; k < size; k++, index++) {
                data[k] += input.data[index];
            }
        }
        div(input.getRow());
    }

    public NNMatrix dot(NNVector vector) {
        NNMatrix result = new NNMatrix(vector.size, size);

        for (int i = 0, index = 0; i < result.getRow(); i++) {
            for (int j = 0; j < result.getColumn(); j++, index++) {
                result.data[index] = vector.data[i] * data[j];
            }
        }

        return result;
    }

    @SneakyThrows
    public void add(NNVector vector) {
        if (size != vector.size) {
            throw new Exception("Vector has difference size");
        }

        if (Use.CPU) {
            for (int i = 0; i < size; i++) {
                data[i] += vector.data[i];
            }
        }

        if (Use.GPU) {
            int n = size;
            CUfunction function = new CUfunction();
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "matAdd_");
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "matAdd_TYPE_");
            }
            Pointer kernelParameters = Pointer.to(Pointer.to(this.data_gpu), Pointer.to(vector.data_gpu), Pointer.to(new int[]{n}));
            int blockSize = Math.min(n, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) n / blockSize);
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                if (!isTYPE()) {
                    IsNan_float();
                    IsNan_float(vector);
                }
                else
                {
                    IsNan();
                    IsNan(vector);
                }
            }
        }
    }

    @SneakyThrows
    public void set(NNVector vector) {
        if (size != vector.size) {
            throw new Exception("Vector has difference size");
        }
        System.arraycopy(vector.data, 0, data, 0, size);
    }

    @SneakyThrows
    public void addProduct(NNVector vector1, NNVector vector2) {
        if (size != vector1.size || size != vector2.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] += vector1.data[i] * vector2.data[i];
        }
    }

    @SneakyThrows
    public void add(NNTensor tensor) {
        if (size != tensor.getDepth()) {
            throw new Exception("Vector has difference size");
        }
        int index = 0;
        for (int i = 0; i < tensor.getRows(); i++) {
            for (int j = 0; j < tensor.getColumns(); j++) {
                for (int k = 0; k < tensor.getDepth(); k++, index++) {
                    data[k] += tensor.data[index];
                }
            }
        }
    }

    @SneakyThrows
    public void add(NNMatrix matrix) {
        if (size != matrix.getColumn()) {
            throw new Exception("Vector has difference size");
        }

        if (Use.CPU) {
            int index = 0;
            for (int i = 0; i < matrix.getRow(); i++) {
                for (int k = 0; k < matrix.getColumn(); k++, index++) {
                    data[k] += matrix.data[index];
                }
            }
        }

        if (Use.GPU) {
            int row = matrix.getRow();
            int column = matrix.getColumn();
            CUfunction function = new CUfunction();
            if (!isTYPE()) {
                cuModuleGetFunction(function, helperModule, "addMatrix");
            }
            else
            {
                cuModuleGetFunction(function, helperModule, "addMatrix_TYPE");
            }
            Pointer kernelParameters = Pointer.to(Pointer.to(matrix.data_gpu), Pointer.to(data_gpu), Pointer.to(new int[]{row}), Pointer.to(new int[]{column}));
            int blockSizeX = (int) Math.min(column, BLOCK_SIZE);
            int gridSizeX = (int) Math.ceil((double) column / blockSizeX);

            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSizeX, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();

                if (!isTYPE()) {
                    matrix.IsNan_float(matrix);
                    IsNan_float();
                }
                else
                {
                    matrix.IsNan(matrix);
                    IsNan();
                }
            }
        }
    }

    @SneakyThrows
    public void subOneDiv(NNVector vector) {
        if (size != vector.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] -= 1.0 / (vector.data[i] + 0.00000001f);
        }
    }

    public void save(FileWriter writer) throws IOException {
        float[] hostData = null;
        short[] hostData_TYPE = null;
        if (Use.GPU) {
            if (!TYPE) {
                hostData = GetFirstSingleValueFloat(data_gpu, size);
            }
            else
            {
                hostData_TYPE = GetAllTYPEValues(data_gpu, size);
            }
        }

        writer.write(TYPE + "\n");
        writer.write(size + "\n");
        for (int i = 0; i < size; i++) {
            if (Use.CPU) {
                writer.write(data[i] + " ");
            }
            else
            {
                if (!TYPE) {
                    assert hostData != null;
                    writer.write(hostData[i] + " ");
                }
                else
                {
                    assert hostData_TYPE != null;
                    writer.write(hostData_TYPE[i] + " ");
                }
            }
            if (i % 1000 == 0) {
                writer.flush();
            }
        }
        writer.write("\n");
        writer.flush();
    }

    public static short[] toShortArray(double[] arr) {
        if (arr == null) return null;
        int n = arr.length;
        short[] ret = new short[n];
        for (int i = 0; i < n; i++) {
            ret[i] = (short) arr[i];
        }
        return ret;
    }

    public static float[] toFloatArray(double[] arr) {
        if (arr == null) return null;
        int n = arr.length;
        float[] ret = new float[n];
        for (int i = 0; i < n; i++) {
            ret[i] = (float) arr[i];
        }
        return ret;
    }

    public static NNVector read(Scanner scanner) {
        boolean TYPE = Boolean.parseBoolean(scanner.nextLine());
        NNVector vector = new NNVector(Integer.parseInt(scanner.nextLine()), TYPE);
        if (Use.CPU) {
            double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
            for (int j = 0; j < vector.size; j++) {
                vector.data[j] = (float) arr[j];
            }
        }
        else {
            if (!vector.TYPE) {
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
                cudaMemcpy(vector.data_gpu, Pointer.to(toFloatArray(arr)), (long) Sizeof.FLOAT * vector.size, cudaMemcpyHostToDevice);
            }
            else
            {
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Short::parseShort).toArray();
                cudaMemcpy(vector.data_gpu, Pointer.to(toShortArray(arr)), (long) Sizeof.SHORT * vector.size, cudaMemcpyHostToDevice);
            }
        }

        return vector;
    }
}
