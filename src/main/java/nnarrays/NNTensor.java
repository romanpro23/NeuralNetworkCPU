package nnarrays;

import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.driver.JCudaDriver;
import jcuda.jcublas.JCublas2;
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
import static jcuda.runtime.JCuda.cudaMemcpy;
import static jcuda.runtime.cudaMemcpyKind.cudaMemcpyHostToDevice;
import static utilities.GPUInit.*;
import static utilities.Use.GPU_Sleep;
import static utilities.Use.GPU_WakeUp;

public class NNTensor extends NNArray {
    @Getter
    private final int depth;
    @Getter
    private final int columns;
    @Getter
    private final int rows;
    @Getter
    private int[] rowsIndex;
    @Getter
    private int[] columnsIndex;

    public NNTensor(int rows, int columns, int depth) {
        super(depth * columns * rows);
        this.depth = depth;
        this.columns = columns;
        this.rows = rows;
        countAxes = 3;

        initialize();
    }

    public NNTensor(int rows, int columns, int depth, boolean TYPE) {
        super(depth * columns * rows, TYPE);
        this.depth = depth;
        this.columns = columns;
        this.rows = rows;
        countAxes = 3;

        initialize();
    }

    public NNTensor(int[] size) {
        this(size[0], size[1], size[2]);
    }

    private void initialize() {
        if (Use.CPU) {
            rowsIndex = new int[rows];
            columnsIndex = new int[columns];
            int sq = depth * columns;
            for (int i = 0; i < rows; i++) {
                rowsIndex[i] = i * sq;
            }
            for (int i = 0; i < columns; i++) {
                columnsIndex[i] = i * depth;
            }
        }
    }

    public NNTensor(int rows, int columns, int depth, float[] data, short[] sdata) {
        super(data, sdata);
        this.depth = depth;
        this.columns = columns;
        this.rows = rows;
        countAxes = 3;

        initialize();
    }

    public NNTensor(int rows, int columns, int depth, float[] data, short[] sdata, boolean TYPE) {
        super(data, sdata, TYPE);
        this.depth = depth;
        this.columns = columns;
        this.rows = rows;
        countAxes = 3;

        initialize();
    }

    @Override
    public int[] shape() {
        return new int[]{rows, columns, depth};
    }

    public float get(int i, int j, int k) {
        return data[rowsIndex[i] + columnsIndex[j] + k];
    }

    public void set(int i, int j, int k, float value) {
        if (Use.CPU) {
            data[rowsIndex[i] + columnsIndex[j] + k] = value;
            sdata[rowsIndex[i] + columnsIndex[j] + k] = floatToBFloat16(value);
        }

        if (Use.GPU) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "set2");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new int[]{i}), Pointer.to(new int[]{j}), Pointer.to(new int[]{k}), Pointer.to(new int[]{columns}), Pointer.to(new int[]{depth}), Pointer.to(new short[]{floatToBFloat16(value)}));
            int blockSize = 1;
            int gridSizeX = 1;
            cuLaunchKernel(function,
                    gridSizeX, 1, 1,      // Grid dimension
                    blockSize, 1, 1,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }
        }
    }

    public NNTensor mul(NNVector vector) {
        NNTensor output = new NNTensor(this.shape());

        for (int i = 0, index = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    output.data[index] = data[index] * vector.data[k];
                }
            }
        }

        return output;
    }

    public NNVector mul(NNTensor tensor) {
        NNVector output = new NNVector(depth);

        for (int i = 0, index = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    output.data[k] += tensor.data[index] * data[index];
                }
            }
        }

        return output;
    }

    public NNTensor spatialMul(NNTensor tensor) {
        NNTensor output = new NNTensor(rows, columns, depth);

        for (int i = 0, index = 0, indexT = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexT++) {
                for (int k = 0; k < depth; k++, index++) {
                    output.data[index] = data[index] * tensor.data[indexT];
                }
            }
        }

        return output;
    }

    public void ClearCpuData()
    {
        data = null;
        sdata = null;
    }

    public NNTensor reverse() {
        if (Use.CPU) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < columns; j++) {
                    for (int k = 0; k < depth; k++) {
                        float val = get(i, j, k);
                        set(i, j, k, get(rows - 1 - i, j, k));
                        set(rows - 1 - i, j, k, val);
                    }
                }
            }
        }

        if (Use.GPU) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "reverse");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(new int[]{rows}), Pointer.to(new int[]{columns}), Pointer.to(new int[]{depth}));
            int blockSizeX = (int) Math.min(rows, Math.pow(BLOCK_SIZE, (double) 1 / 2.5));
            int blockSizeY = (int) Math.min(columns, Math.pow(BLOCK_SIZE, (double) 1 / 2.5));
            int blockSizeZ = (int) Math.min(depth, Math.pow(BLOCK_SIZE, (double) 1 / 5));
            int gridSizeX = (int) Math.ceil((double) rows / blockSizeX);
            int gridSizeY = (int) Math.ceil((double) columns / blockSizeY);
            int gridSizeZ = (int) Math.ceil((double) depth / blockSizeZ);

            cuLaunchKernel(function,
                    gridSizeX, gridSizeY, gridSizeZ,      // Grid dimension
                    blockSizeX, blockSizeY, blockSizeZ,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );

            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan();
            }
        }
        return this;
    }

    public void prelu(NNTensor input, NNVector alpha) {
        for (int i = 0, index = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    if (input.data[index] < 0) {
                        data[index] = input.data[index] * alpha.data[k];
                    } else {
                        data[index] = input.data[index];
                    }
                }
            }
        }
    }

    public void derPrelu(NNTensor input, NNTensor error, NNVector alpha) {
        for (int i = 0, index = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    if (input.data[index] < 0) {
                        data[index] = error.data[index] * alpha.data[k];
                    } else {
                        data[index] = error.data[index];
                    }
                }
            }
        }
    }

    public void shuffle(NNTensor input, int countGroup) {
        int sizeGroup = input.depth / countGroup;
        int outIndex, inIndex;

        int[] index = new int[countGroup];
        for (int i = 0; i < countGroup; i++) {
            index[i] = i * sizeGroup;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                outIndex = rowsIndex[i] + columnsIndex[j];
                inIndex = input.rowsIndex[i] + input.columnsIndex[j];
                for (int k = 0, d = 0; k < sizeGroup; k++, inIndex++) {
                    for (int l = 0; l < countGroup; l++, d++, outIndex++) {
                        data[outIndex] = input.data[inIndex + index[l]];
                    }
                }
            }
        }
    }

    public void backShuffle(NNTensor input, int countGroup) {
        int sizeGroup = input.depth / countGroup;

        int outIndex, inIndex;

        int[] index = new int[countGroup];
        for (int i = 0; i < countGroup; i++) {
            index[i] = i * sizeGroup;
        }

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                outIndex = rowsIndex[i] + columnsIndex[j];
                inIndex = input.rowsIndex[i] + input.columnsIndex[j];
                for (int k = 0, d = 0; k < sizeGroup; k++, outIndex++) {
                    for (int l = 0; l < countGroup; l++, d++, inIndex++) {
                        data[outIndex + index[l]] = input.data[inIndex];
                    }
                }
            }
        }
    }

    public void maxPool(NNTensor input, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        int x0, y0, inputIndex, outputIndex, outIndex;
        Arrays.fill(data, -1000);

        for (int y = -paddingY, h = 0; h < rows; y += step, h++) {
            for (int x = -paddingX, w = 0; w < columns; x += step, w++) {
                outputIndex = rowsIndex[h] + columnsIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.rows) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.columns) {
                            continue;
                        }
                        inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < depth; d++, inputIndex++, outIndex++) {
                            float val = input.data[inputIndex];

                            if (data[outIndex] < val) {
                                data[outIndex] = val;
                            }
                        }
                    }
                }
            }
        }
    }

    public void upSampling(NNTensor input, int heightKernel, int widthKernel) {
        int xIndex, yIndex, inputIndex, outputIndex, inIndex;

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.columns; x++) {
                inIndex = input.rowsIndex[y] + input.columnsIndex[x];
                for (int j = 0; j < heightKernel; j++) {
                    yIndex = y * heightKernel + j;
                    for (int k = 0; k < widthKernel; k++) {
                        xIndex = x * widthKernel + k;
                        outputIndex = rowsIndex[yIndex] + columnsIndex[xIndex];
                        inputIndex = inIndex;
                        for (int d = 0; d < depth; d++, outputIndex++, inputIndex++) {
                            data[outputIndex] = input.data[inputIndex];
                        }
                    }
                }
            }
        }
    }

    public void pixelShuffle(NNTensor input, int sizeKernel) {
        int xIndex, yIndex;

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.columns; x++) {
                for (int d = 0, depthIn = 0; d < depth; d++) {
                    for (int j = 0; j < sizeKernel; j++) {
                        yIndex = y * sizeKernel + j;
                        for (int k = 0; k < sizeKernel; k++, depthIn++) {
                            xIndex = x * sizeKernel + k;
                            set(yIndex, xIndex, d, input.get(y, x, depthIn));
                        }
                    }
                }
            }
        }
    }

    public void backPixelShuffle(NNTensor input, int sizeKernel) {
        int xIndex, yIndex;

        for (int y = 0; y < input.rows; y++) {
            for (int x = 0; x < input.columns; x++) {
                for (int d = 0, depthIn = 0; d < depth; d++) {
                    for (int j = 0; j < sizeKernel; j++) {
                        yIndex = y * sizeKernel + j;
                        for (int k = 0; k < sizeKernel; k++, depthIn++) {
                            xIndex = x * sizeKernel + k;
                            set(y, x, depthIn, input.get(yIndex, xIndex, d));
                        }
                    }
                }
            }
        }
    }

    public void backUpSampling(NNTensor input, int heightKernel, int widthKernel) {
        int xIndex, yIndex, inputIndex, outputIndex, outIndex;

        for (int y = 0; y < rows; y++) {
            for (int x = 0; x < columns; x++) {
                outIndex = rowsIndex[y] + columnsIndex[x];
                for (int j = 0; j < heightKernel; j++) {
                    yIndex = y * heightKernel + j;
                    for (int k = 0; k < widthKernel; k++) {
                        outputIndex = outIndex;
                        xIndex = x * widthKernel + k;
                        inputIndex = input.rowsIndex[yIndex] + input.columnsIndex[xIndex];
                        for (int d = 0; d < depth; d++, outputIndex++, inputIndex++) {
                            data[outputIndex] += input.data[inputIndex];
                        }
                    }
                }
            }
        }

        div(heightKernel * widthKernel);
    }

    public void averagePool(NNTensor input, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        int x0, y0, inputIndex, outputIndex, outIndex;

        for (int y = -paddingY, h = 0; h < rows; y += step, h++) {
            for (int x = -paddingX, w = 0; w < columns; x += step, w++) {
                outputIndex = rowsIndex[h] + columnsIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.rows) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.columns) {
                            continue;
                        }
                        inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < depth; d++, inputIndex++, outIndex++) {
                            data[outIndex] += input.data[inputIndex];
                        }
                    }
                }
            }
        }
    }

    public void backMaxPool(NNTensor error, NNTensor input, NNTensor outputs,
                            int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        int x0, y0, inputIndex, outputIndex, outIndex;

        for (int y = -paddingY, h = 0; h < outputs.rows; y += step, h++) {
            for (int x = -paddingX, w = 0; w < outputs.columns; x += step, w++) {
                outputIndex = outputs.rowsIndex[h] + outputs.columnsIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.rows) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.columns) {
                            continue;
                        }
                        inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < depth; d++, inputIndex++, outIndex++) {
                            if (input.data[inputIndex] == outputs.data[outIndex]) {
                                this.data[inputIndex] += error.data[outIndex];
                            }
                        }
                    }
                }
            }
        }
    }

    public void backAveragePool(NNTensor error, NNTensor input, NNTensor outputs,
                                int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        int x0, y0, inputIndex, outputIndex, outIndex;
        error.div(heightKernel * widthKernel);

        for (int y = -paddingY, h = 0; h < outputs.rows; y += step, h++) {
            for (int x = -paddingX, w = 0; w < outputs.columns; x += step, w++) {
                outputIndex = outputs.rowsIndex[h] + outputs.columnsIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.rows) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.columns) {
                            continue;
                        }
                        inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < input.depth; d++, inputIndex++, outIndex++) {
                            this.data[inputIndex] += error.data[outIndex];
                        }
                    }
                }
            }
        }
    }

    public void backGlobalMaxPool(NNTensor input, NNVector output, NNVector error) {
        int index = 0;
        for (int i = 0; i < input.getRows(); i++) {
            for (int j = 0; j < input.getColumns(); j++) {
                for (int k = 0; k < input.getDepth(); k++, index++) {
                    if (output.data[k] == input.data[index]) {
                        data[index] += error.data[k];
                    }
                }
            }
        }
    }

    public void backGlobalAveragePool(NNVector error) {
        int index = 0;
        error.div(rows * columns);
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                for (int k = 0; k < depth; k++, index++) {
                    data[index] += error.data[k];
                }
            }
        }
    }

    public NNTensor spatialAveragePool() {
        NNTensor result = new NNTensor(rows, columns, 1);
        int index = 0, indexOut = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexOut++) {
                for (int k = 0; k < depth; k++, index++) {
                    result.data[indexOut] += data[index];
                }
                result.data[indexOut] /= depth;
            }
        }
        return result;
    }

    public NNTensor backSpatialMul() {
        NNTensor result = new NNTensor(rows, columns, 1);
        int index = 0, indexOut = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexOut++) {
                for (int k = 0; k < depth; k++, index++) {
                    result.data[indexOut] += data[index];
                }
            }
        }
        return result;
    }

    public NNMatrix softmaxSum(NNMatrix softmax) {
        NNMatrix result = new NNMatrix(rows, depth);

        int index = 0, indexOut;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float c = softmax.get(i, j);
                indexOut = result.getRowIndex()[i];
                for (int k = 0; k < depth; k++, index++, indexOut++) {
                    result.data[indexOut] += data[index] * c;
                }
            }
        }
        return result;
    }

    public NNTensor backSoftmaxSum(NNMatrix softmax, NNMatrix error) {
        NNTensor result = new NNTensor(rows, columns, depth);

        int index = 0, indexOut;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                float c = softmax.get(i, j);
                indexOut = error.getRowIndex()[i];
                for (int k = 0; k < depth; k++, index++, indexOut++) {
                    result.data[index] += error.data[indexOut] * c;
                }
            }
        }
        return result;
    }

    public NNTensor spatialMaxPool() {
        NNTensor result = new NNTensor(rows, columns, 1);
        result.fill(Float.MIN_VALUE);
        int index = 0, indexOut = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexOut++) {
                for (int k = 0; k < depth; k++, index++) {
                    if (result.data[indexOut] < data[index]) {
                        result.data[indexOut] = data[index];
                    }
                }
            }
        }
        return result;
    }

    public NNTensor backSpatialMaxPool(NNTensor max, NNTensor error) {
        NNTensor result = new NNTensor(rows, columns, depth);
        int index = 0, indexOut = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexOut++) {
                for (int k = 0; k < depth; k++, index++) {
                    if (data[index] == max.data[indexOut]) {
                        result.data[index] = error.data[indexOut];
                    }
                }
            }
        }
        return result;
    }

    public NNTensor backSpatialAveragePool(NNTensor error) {
        NNTensor result = new NNTensor(rows, columns, depth);
        error.div(depth);
        int index = 0, indexOut = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexOut++) {
                for (int k = 0; k < depth; k++, index++) {
                    result.data[index] = error.data[indexOut];
                }
            }
        }
        return result;
    }

    public NNTensor concat(NNTensor tensor) {
        NNTensor result = new NNTensor(rows, columns, depth + tensor.depth);
        int index1, index2, indexOut;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                index1 = rowsIndex[i] + columnsIndex[j];
                index2 = tensor.rowsIndex[i] + tensor.columnsIndex[j];
                indexOut = result.rowsIndex[i] + result.columnsIndex[j];
                System.arraycopy(data, index1, result.data, indexOut, depth);
                indexOut = result.rowsIndex[i] + result.columnsIndex[j] + depth;
                System.arraycopy(tensor.data, index2, result.data, indexOut, tensor.depth);
            }
        }

        return result;
    }

    public NNTensor subFlatTensor(int index) {
        NNTensor result = new NNTensor(rows, columns, 1);
        int indexOut = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++, indexOut++) {
                result.data[indexOut] = get(i, j, index);
            }
        }

        return result;
    }

    public void convolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        float val;

        for (int d = 0; d < weight.depth(); d++) {
            for (int y = -padY, h = 0; h < rows; y += step, h++) {
                for (int x = -padX, w = 0; w < columns; x += step, w++) {
                    outputIndex = rowsIndex[h] + columnsIndex[w] + d;
                    val = 0;
                    for (int j = 0; j < weight.length(); j++) {
                        y0 = y + j;
                        if (y0 < 0 || y0 >= input.rows) {
                            continue;
                        }
                        w0 = weight.depthIndex()[d] + weight.lengthIndex()[j];
                        for (int k = 0; k < weight.row(); k++) {
                            x0 = x + k;
                            if (x0 < 0 || x0 >= input.columns) {
                                continue;
                            }
                            inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                            weightIndex = w0 + weight.rowIndex()[k];
                            for (int c = 0; c < weight.column(); c++, inputIndex++, weightIndex++) {
                                val += input.data[inputIndex] * weight.data[weightIndex];
                            }
                        }
                    }
                    data[outputIndex] = val;
                }
            }
        }
    }

    public NNMatrix imageVector(int sizeKernel) {
        int row = (rows / sizeKernel) * (columns / sizeKernel);
        int col = sizeKernel * sizeKernel * depth;
        NNMatrix result = new NNMatrix(row, col);

        if (Use.CPU) {
            int index = 0, indexInput;

            for (int h = 0; h < rows; h += sizeKernel) {
                for (int w = 0; w < columns; w += sizeKernel) {
                    for (int j = 0; j < sizeKernel; j++) {
                        for (int k = 0; k < sizeKernel; k++) {
                            indexInput = rowsIndex[h + j] + columnsIndex[w + k];
                            for (int c = 0; c < depth; c++, index++, indexInput++) {
                                result.data[index] = data[indexInput];
                            }
                        }
                    }
                }
            }
        }

        if (Use.GPU) {
            int Rows = rows;
            int Cols = columns;
            //long start0 = System.nanoTime();
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "imageVector");
            Pointer kernelParameters = Pointer.to(Pointer.to(data_gpu), Pointer.to(result.data_gpu),  Pointer.to(new int[]{Rows}), Pointer.to(new int[]{Cols}), Pointer.to(new int[]{depth}), Pointer.to(new int[]{sizeKernel}));
            int blockSizeX = (int) Math.min((double) Rows / sizeKernel, Math.pow(BLOCK_SIZE, (double) 1 / 2.5));
            int blockSizeY = (int) Math.min((double) Cols / sizeKernel, Math.pow(BLOCK_SIZE, (double) 1 / 2.5));
            int blockSizeZ = (int) Math.min(sizeKernel, Math.pow(BLOCK_SIZE, (double) 1 / 5));
            int gridSizeX = (int) Math.ceil((double) Rows / blockSizeX / sizeKernel);
            int gridSizeY = (int) Math.ceil((double) Cols / blockSizeY / sizeKernel);
            int gridSizeZ = (int) Math.ceil((double) sizeKernel / blockSizeZ);

            cuLaunchKernel(function,
                    gridSizeX, gridSizeY, gridSizeZ,      // Grid dimension
                    blockSizeX, blockSizeY, blockSizeZ,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(result);
                IsNan();
            }
            //System.out.println(" ! " + (System.nanoTime() - start0) / 1000 + " ! ");
        }
        return result;
    }

    public NNTensor backImageVector(NNMatrix error ,int sizeKernel) {
        NNTensor result = new NNTensor(rows, columns, depth);
        if (Use.CPU) {
            int index = 0, indexInput;

            for (int h = 0; h < rows; h += sizeKernel) {
                for (int w = 0; w < columns; w += sizeKernel) {
                    for (int j = 0; j < sizeKernel; j++) {
                        for (int k = 0; k < sizeKernel; k++) {
                            indexInput = rowsIndex[h + j] + columnsIndex[w + k];
                            for (int c = 0; c < depth; c++, index++, indexInput++) {
                                result.data[indexInput] = error.data[index];
                            }
                        }
                    }
                }
            }
        }

        if (Use.GPU) {
            CUfunction function = new CUfunction();
            cuModuleGetFunction(function, helperModule, "backImageVector");
            Pointer kernelParameters = Pointer.to(Pointer.to(error.data_gpu), Pointer.to(result.data_gpu),  Pointer.to(new int[]{rows}), Pointer.to(new int[]{columns}), Pointer.to(new int[]{depth}), Pointer.to(new int[]{sizeKernel}));
            int blockSizeX = (int) Math.min(rows, Math.pow(BLOCK_SIZE, (double) 1 / 2.5));
            int blockSizeY = (int) Math.min(columns, Math.pow(BLOCK_SIZE, (double) 1 / 2.5));
            int blockSizeZ = (int) Math.min(sizeKernel, Math.pow(BLOCK_SIZE, (double) 1 / 5));
            int gridSizeX = (int) Math.ceil((double) rows / blockSizeX);
            int gridSizeY = (int) Math.ceil((double) columns / blockSizeY);
            int gridSizeZ = (int) Math.ceil((double) sizeKernel / blockSizeZ);

            cuLaunchKernel(function,
                    gridSizeX, gridSizeY, gridSizeZ,      // Grid dimension
                    blockSizeX, blockSizeY, blockSizeZ,      // Block dimension
                    0, null,               // Shared memory size and stream
                    kernelParameters, null // Kernel- and extra parameters
            );
            if (Use.DEBUG_SYNC) {
                JCudaDriver.cuCtxSynchronize();
                IsNan(error);
                IsNan(result);
            }
        }
        return result;
    }

    public NNMatrix convolutionImg2Col(NNTensor input, NNTensor4D weight, int step, int padY, int padX) {
        data = null;
        NNMatrix weightM = new NNMatrix(weight.depth(), weight.size / weight.depth(), weight.getData(), weight.getSdata());
        NNMatrix inputM = img2col(input, weight, step, padY, padX);
        data = inputM.dotT(weightM).data;
        return inputM;
    }

    public NNMatrix img2col(NNTensor input, NNTensor4D weight, int step, int padY, int padX) {
        NNMatrix result = new NNMatrix(rows * columns, weight.size / weight.depth());
        int x0, y0, inputIndex;

        for (int y = -padY, out = 0, h = 0; h < rows; y += step, h++) {
            for (int x = -padX, w = 0; w < columns; x += step, w++) {
                for (int j = 0; j < weight.length(); j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.rows) {
                        out += weight.row() * weight.column();
                        continue;
                    }
                    for (int k = 0; k < weight.row(); k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.columns) {
                            out += weight.column();
                            continue;
                        }
                        inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                        for (int c = 0; c < weight.column(); c++, inputIndex++, out++) {
                            result.data[out] = input.data[inputIndex];
                        }
                    }
                }
            }
        }
        return result;
    }

    public NNTensor deImg2col(NNTensor input, NNMatrix error, NNTensor4D weight, int step, int padY, int padX) {
        NNTensor result = new NNTensor(input.shape());
        int x0, y0, inputIndex;

        for (int y = -padY, out = 0, h = 0; h < rows; y += step, h++) {
            for (int x = -padX, w = 0; w < columns; x += step, w++) {
                for (int j = 0; j < weight.length(); j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.rows) {
                        out += weight.row() * weight.column();
                        continue;
                    }
                    for (int k = 0; k < weight.row(); k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.columns) {
                            out += weight.column();
                            continue;
                        }
                        inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                        for (int c = 0; c < weight.column(); c++, inputIndex++, out++) {
                            result.data[inputIndex] = error.data[out];
                        }
                    }
                }
            }
        }
        return result;
    }

    public void deformableConvolution(NNTensor input, NNTensor offset, int heightKernel, int widthKernel, int step,
                                      int padY, int padX) {
        float x0, y0;
        int x0_l, y0_l, x0_h, y0_h;
        float lh, lw, hh, hw;

        float v4, v1, v2, v3;

        for (int y = -padY, h = 0; h < offset.rows; y += step, h++) {
            for (int x = -padX, w = 0; w < offset.columns; x += step, w++) {
                for (int j = 0, offst = 0; j < heightKernel; j++) {
                    y0 = y + j + offset.get(h, w, offst);
                    y0_l = (int) y0;
                    y0_h = y0_l + 1;
                    if (y0_h < 0 || y0_l >= input.rows) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++, offst += 2) {
                        x0 = x + k + offset.get(h, w, offst + 1);
                        x0_l = (int) x0;
                        x0_h = x0_l + 1;
                        if (x0_h < 0 || x0_l >= input.columns) {
                            continue;
                        }

                        lh = y0 - y0_l;
                        hh = 1.0f - lh;
                        lw = x0 - x0_l;
                        hw = 1.0f - lw;

                        float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

                        for (int c = 0; c < input.depth; c++) {
                            v4 = v1 = v2 = v3 = 0;
                            if (y0_l >= 0 && x0_l >= 0)
                                v1 = input.get(y0_l, x0_l, c);
                            if (y0_l >= 0 && x0_h < input.columns)
                                v2 = input.get(y0_l, x0_h, c);
                            if (y0_h < input.rows && x0_l >= 0)
                                v3 = input.get(y0_h, x0_l, c);
                            if (y0_h < input.rows && x0_h < input.columns)
                                v4 = input.get(y0_h, x0_h, c);

                            set(h * heightKernel + j, w * widthKernel + k, c, (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4));
                        }
                    }
                }
            }
        }
    }

    public void modulatedConvolution(NNTensor offset, NNTensor mask, int heightKernel, int widthKernel) {
        int maskIndex, height = offset.rows / heightKernel, width = offset.columns / widthKernel;
        int index;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int j = 0, offst = 0; j < heightKernel; j++) {
                    for (int k = 0; k < widthKernel; k++, offst++) {
                        maskIndex = mask.rowsIndex[h] + mask.columnsIndex[w] + offst;
                        index = offset.rowsIndex[h * heightKernel + j] + offset.columnsIndex[w * widthKernel + k];
                        for (int c = 0; c < offset.depth; c++, index++) {
                            data[index] = mask.data[maskIndex] * offset.data[index];
                        }
                    }
                }
            }
        }
    }

    public void backModulatedConvolution(NNTensor offset, NNTensor error, int heightKernel, int widthKernel) {
        int maskIndex, height = offset.rows / heightKernel, width = offset.columns / widthKernel;
        int index;

        for (int h = 0; h < height; h++) {
            for (int w = 0; w < width; w++) {
                for (int j = 0, offst = 0; j < heightKernel; j++) {
                    for (int k = 0; k < widthKernel; k++, offst++) {
                        maskIndex = rowsIndex[h] + columnsIndex[w] + offst;
                        index = offset.rowsIndex[h * heightKernel + j] + offset.columnsIndex[w * widthKernel + k];
                        for (int c = 0; c < offset.depth; c++, index++) {
                            data[maskIndex] += error.data[index] * offset.data[index];
                        }
                    }
                }
            }
        }
    }

    public void backDeformableConvolution(NNTensor input, NNTensor offset, NNTensor offset_error, NNTensor
            offset_delta,
                                          int heightKernel, int widthKernel, int step, int padY, int padX) {
        float x0, y0;
        int x0_l, y0_l, x0_h, y0_h;
        float lh, lw, hh, hw;

        float v4, v1, v2, v3;
        float error;

        for (int y = -padY, h = 0; h < offset.rows; y += step, h++) {
            for (int x = -padX, w = 0; w < offset.columns; x += step, w++) {
                for (int j = 0, offst = 0; j < heightKernel; j++) {
                    y0 = y + j + offset.get(h, w, offst);
                    y0_l = (int) y0;
                    y0_h = y0_l + 1;
                    if (y0_h < 0 || y0_l >= input.rows) {
                        continue;
                    }
                    lh = y0 - y0_l;
                    hh = 1.0f - lh;
                    for (int k = 0; k < widthKernel; k++, offst += 2) {
                        x0 = x + k + offset.get(h, w, offst + 1);
                        x0_l = (int) x0;
                        x0_h = x0_l + 1;
                        if (x0_h < 0 || x0_l >= input.columns) {
                            continue;
                        }
                        lw = x0 - x0_l;
                        hw = 1.0f - lw;

                        float w1 = hh * hw, w2 = hh * lw, w3 = lh * hw, w4 = lh * lw;

                        for (int c = 0; c < input.depth; c++) {
                            error = offset_error.get(h * heightKernel + j, w * widthKernel + k, c);
                            if (y0_l >= 0 && x0_l >= 0) {
                                v1 = input.get(y0_l, x0_l, c) * error;
                                add(y0_l, x0_l, c, w1 * error);
                                offset_delta.add(h, w, offst, -1 * hw * v1);
                                offset_delta.add(h, w, offst + 1, -1 * hh * v1);
                            }
                            if (y0_l >= 0 && x0_h < input.columns) {
                                v2 = input.get(y0_l, x0_h, c) * error;
                                add(y0_l, x0_h, c, w2 * error);
                                offset_delta.add(h, w, offst, -1 * lw * v2);
                                offset_delta.add(h, w, offst + 1, hh * v2);
                            }
                            if (y0_h < input.rows && x0_l >= 0) {
                                v3 = input.get(y0_h, x0_l, c) * error;
                                add(y0_h, x0_l, c, w3 * error);
                                offset_delta.add(h, w, offst, hw * v3);
                                offset_delta.add(h, w, offst + 1, -1 * lh * v3);
                            }
                            if (y0_h < input.rows && x0_h < input.columns) {
                                v4 = input.get(y0_h, x0_h, c) * error;
                                add(y0_h, x0_h, c, w4 * error);
                                offset_delta.add(h, w, offst, lw * v4);
                                offset_delta.add(h, w, offst + 1, lh * v4);
                            }
                        }
                    }
                }
            }
        }
    }

    public void dilatedConvolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX, int dilatationY,
                                   int dilatationX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        float val;

        int dilY = (((weight.length() - 1) * dilatationY + 1) - weight.length()) / 2;
        int dilX = (((weight.row() - 1) * dilatationX + 1) - weight.row()) / 2;

        for (int y = -padY - dilY, h = 0; h < rows; y += step, h++) {
            for (int x = -padX - dilX, w = 0; w < columns; x += step, w++) {
                outputIndex = rowsIndex[h] + columnsIndex[w];
                for (int d = 0; d < weight.depth(); d++, outputIndex++) {
                    val = 0;
                    for (int j = 0; j < weight.length(); j++) {
                        y0 = y + j * dilatationY;
                        if (y0 < 0 || y0 >= input.rows) {
                            continue;
                        }
                        w0 = weight.depthIndex()[d] + weight.lengthIndex()[j];
                        for (int k = 0; k < weight.row(); k++) {
                            x0 = x + k * dilatationX;
                            if (x0 < 0 || x0 >= input.columns) {
                                continue;
                            }
                            inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                            weightIndex = w0 + weight.rowIndex()[k];
                            for (int c = 0; c < weight.column(); c++, inputIndex++, weightIndex++) {
                                val += input.data[inputIndex] * weight.data[weightIndex];
                            }
                        }
                    }
                    data[outputIndex] = val;
                }
            }
        }
    }

    public void groupConvolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX, int countGroup) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        float val;
        int sizeGroupKernel = weight.depth() / countGroup;

        for (int g = 0, gI, gO; g < countGroup; g++) {
            gI = g * weight.column();
            gO = g * sizeGroupKernel;
            for (int y = -padY, h = 0; h < rows; y += step, h++) {
                for (int x = -padX, w = 0; w < columns; x += step, w++) {
                    outputIndex = rowsIndex[h] + columnsIndex[w] + gO;
                    for (int d = 0; d < sizeGroupKernel; d++, outputIndex++) {
                        val = 0;
                        for (int j = 0; j < weight.length(); j++) {
                            y0 = y + j;
                            if (y0 < 0 || y0 >= input.rows) {
                                continue;
                            }
                            w0 = weight.depthIndex()[gO + d] + weight.lengthIndex()[j];
                            for (int k = 0; k < weight.row(); k++) {
                                x0 = x + k;
                                if (x0 < 0 || x0 >= input.columns) {
                                    continue;
                                }
                                inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0] + gI;
                                weightIndex = w0 + weight.rowIndex()[k];
                                for (int c = 0; c < weight.column(); c++, weightIndex++, inputIndex++) {
                                    val += input.data[inputIndex] * weight.data[weightIndex];
                                }
                            }
                        }
                        data[outputIndex] = val;
                    }
                }
            }
        }
    }

    public void convolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX, int countGroup,
                            int dilY, int dilX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        float val;
        final int sizeGroupKernel = weight.depth() / countGroup;
        final int _dilY = (((weight.length() - 1) * dilY + 1) - weight.length()) / 2;
        final int _dilX = (((weight.row() - 1) * dilX + 1) - weight.row()) / 2;

        for (int g = 0, gI, gO; g < countGroup; g++) {
            gI = g * weight.column();
            gO = g * sizeGroupKernel;
            for (int d = 0; d < sizeGroupKernel; d++) {
                for (int y = -padY - _dilY, h = 0; h < rows; y += step, h++) {
                    for (int x = -padX - _dilX, w = 0; w < columns; x += step, w++) {
                        outputIndex = rowsIndex[h] + columnsIndex[w] + gO + d;
                        val = 0;
                        for (int j = 0; j < weight.length(); j++) {
                            y0 = y + j * dilY;
                            if (y0 < 0 || y0 >= input.rows) {
                                continue;
                            }
                            w0 = weight.depthIndex()[gO + d] + weight.lengthIndex()[j];
                            for (int k = 0; k < weight.row(); k++) {
                                x0 = x + k * dilX;
                                if (x0 < 0 || x0 >= input.columns) {
                                    continue;
                                }
                                inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0] + gI;
                                weightIndex = w0 + weight.rowIndex()[k];
                                for (int c = 0; c < weight.column(); c++, weightIndex++, inputIndex++) {
                                    val += input.data[inputIndex] * weight.data[weightIndex];
                                }
                            }
                        }
                        data[outputIndex] = val;
                    }
                }
            }
        }
    }

    public void transposeConvolution(NNTensor input, NNTensor4D weight, int stride, int paddingY, int paddingX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        int padY = weight.length() - 1 - paddingY;
        int padX = weight.row() - 1 - paddingX;
        int wCore = weight.row() - 1;
        int hCore = weight.length() - 1;
        int hC, wC;

        float val;

        for (int y = -padY, h = 0; h < rows; y++, h++) {
            for (int x = -padX, w = 0; w < columns; x++, w++) {
                outputIndex = rowsIndex[h] + columnsIndex[w];
                for (int d = 0; d < weight.column(); d++, outputIndex++) {
                    val = 0;
                    for (int j = 0; j < weight.length(); j++) {
                        y0 = y + j;
                        if (y0 < 0 || y0 >= input.rows || (stride > 1 && y0 % stride != 0)) {
                            continue;
                        }
                        hC = hCore - j;
                        w0 = weight.lengthIndex()[hC] + d;
                        for (int k = 0; k < weight.row(); k++) {
                            x0 = x + k;
                            if (x0 < 0 || x0 >= input.columns || (stride > 1 && x0 % stride != 0)) {
                                continue;
                            }
                            inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                            wC = wCore - k;
                            weightIndex = w0 + weight.rowIndex()[wC];

                            for (int c = 0; c < weight.depth(); c++, inputIndex++) {
                                val += input.data[inputIndex] * weight.data[weight.depthIndex()[c] + weightIndex];
                            }
                        }
                    }
                    data[outputIndex] += val;
                }
            }
        }
    }

    public void transposeDilatedConvolution(NNTensor input, NNTensor4D weight, int paddingY, int paddingX,
                                            int dilatationY, int dilatationX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        int padY = weight.length() - 1 - paddingY;
        int padX = weight.row() - 1 - paddingX;
        int wCore = weight.row() - 1;
        int hCore = weight.length() - 1;
        int hC, wC;

        int dilY = (((weight.length() - 1) * dilatationY + 1) - weight.length()) / 2;
        int dilX = (((weight.row() - 1) * dilatationX + 1) - weight.row()) / 2;

        float val;

        for (int y = -padY - dilY, h = 0; h < rows; y++, h++) {
            for (int x = -padX - dilX, w = 0; w < columns; x++, w++) {
                outputIndex = rowsIndex[h] + columnsIndex[w];
                for (int d = 0; d < weight.column(); d++, outputIndex++) {
                    val = 0;
                    for (int j = 0; j < weight.length(); j++) {
                        y0 = y + j * dilatationY;
                        if (y0 < 0 || y0 >= input.rows) {
                            continue;
                        }
                        hC = hCore - j;
                        w0 = weight.lengthIndex()[hC] + d;
                        for (int k = 0; k < weight.row(); k++) {
                            x0 = x + k * dilatationX;
                            if (x0 < 0 || x0 >= input.columns) {
                                continue;
                            }
                            inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0];
                            wC = wCore - k;
                            weightIndex = w0 + weight.rowIndex()[wC];

                            for (int c = 0; c < weight.depth(); c++, inputIndex++) {
                                val += input.data[inputIndex] * weight.data[weight.depthIndex()[c] + weightIndex];
                            }
                        }
                    }
                    data[outputIndex] = val;
                }
            }
        }
    }

    public void transposeGroupConvolution(NNTensor input, NNTensor4D weight, int paddingY, int paddingX,
                                          int countGroup) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        int padY = weight.length() - 1 - paddingY;
        int padX = weight.row() - 1 - paddingX;
        int wCore = weight.row() - 1;
        int hCore = weight.length() - 1;
        int hC, wC;

        float val;

        int sizeGroupKernel = weight.depth() / countGroup;

        for (int g = 0, gI, gO; g < countGroup; g++) {
            gI = g * weight.column();
            gO = g * sizeGroupKernel;
            for (int y = -padY, h = 0; h < rows; y++, h++) {
                for (int x = -padX, w = 0; w < columns; x++, w++) {
                    outputIndex = rowsIndex[h] + columnsIndex[w] + gI;
                    for (int d = 0; d < weight.column(); d++, outputIndex++) {
                        val = 0;
                        for (int j = 0; j < weight.length(); j++) {
                            y0 = y + j;
                            if (y0 < 0 || y0 >= input.rows) {
                                continue;
                            }
                            hC = hCore - j;
                            w0 = weight.lengthIndex()[hC] + d;
                            for (int k = 0; k < weight.row(); k++) {
                                x0 = x + k;
                                if (x0 < 0 || x0 >= input.columns) {
                                    continue;
                                }
                                inputIndex = input.rowsIndex[y0] + input.columnsIndex[x0] + gO;
                                wC = wCore - k;
                                weightIndex = w0 + weight.rowIndex()[wC];

                                for (int c = 0, wg = gO; c < sizeGroupKernel; wg++, c++, inputIndex++) {
                                    val += input.data[inputIndex] * weight.data[weightIndex + weight.getDepthIndex()[wg]];
                                }
                            }
                        }
                        data[outputIndex] = val;
                    }
                }
            }
        }
    }

    public void add(int i, int j, int k, float val) {
        data[rowsIndex[i] + columnsIndex[j] + k] += val;
    }

    @SneakyThrows
    public void add(NNVector vector) {
        if (depth != vector.size) {
            throw new Exception("Array has difference size");
        }
        int inputIndex;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                inputIndex = rowsIndex[i] + columnsIndex[j];
                for (int k = 0; k < depth; k++, inputIndex++) {
                    data[inputIndex] += vector.data[k];
                }
            }
        }
    }

    public void addDerScalarMul(NNMatrix matrix, NNMatrix error) {
        for (int i = 0; i < error.getRow(); i++) {
            for (int j = 0; j < error.getColumn(); j++) {
                for (int k = 0; k < matrix.getColumn(); k++) {
                    add(i, j, k, error.get(i, j) * matrix.get(i, k));
                }
            }
        }
    }

    public NNTensor stride(int stride) {
        if (stride == 1) {
            return this;
        }
        NNTensor result = new NNTensor(rows * stride, columns * stride, depth);
        int inputIndex, outpuIndex, i_s, j_s;
        for (int i = 0; i < rows; i++) {
            i_s = i * stride;
            for (int j = 0; j < columns; j++) {
                j_s = j * stride;
                inputIndex = rowsIndex[i] + columnsIndex[j];
                outpuIndex = result.rowsIndex[i_s] + result.columnsIndex[j_s];
                for (int k = 0; k < depth; k++, inputIndex++, outpuIndex++) {
                    result.data[outpuIndex] = data[inputIndex];
                }
            }
        }
        return result;
    }

    public void convolution(NNMatrix input, NNMatrix error, int step, int pad) {
        int y0, inputIndex, weightIndex, outputIndex;

        for (int y = -pad, h = 0; h < error.getRow(); y += step, h++) {
            outputIndex = error.getRowIndex()[h];
            for (int d = 0; d < rows; d++, outputIndex++) {
                for (int j = 0; j < columns; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.getRow()) {
                        continue;
                    }
                    inputIndex = input.getRowIndex()[y0];
                    weightIndex = rowsIndex[d] + columnsIndex[j];
                    for (int c = 0; c < depth; c++, inputIndex++, weightIndex++) {
                        data[weightIndex] += input.data[inputIndex] * error.data[outputIndex];
                    }
                }
            }
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
        writer.write(rows + " " + columns + " " + depth + "\n");
        for (int d = 0; d < rows; d++) {
            for (int i = 0; i < columns; i++) {
                for (int j = 0; j < depth; j++) {
                    if (Use.CPU) {
                        writer.write(get(d, i, j) + " ");
                    }
                    else
                    {
                        if (!TYPE) {
                            assert hostData != null;
                            writer.write(hostData[d * depth * columns + i * depth + j] + " ");
                        }
                        else
                        {
                            assert hostData_TYPE != null;
                            writer.write(hostData_TYPE[d * depth * columns + i * depth + j] + " ");
                        }
                    }
                }
            }
            writer.write("\n");
            writer.flush();
        }
    }

    public static NNTensor read(Scanner scanner) {
        boolean TYPE = Boolean.parseBoolean(scanner.nextLine());
        int[] size = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        NNTensor tensor = new NNTensor(size[0], size[1], size[2], TYPE);
        if (Use.CPU) {
            int index = 0;
            for (int d = 0; d < tensor.rows; d++) {
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
                for (double v : arr) {
                    tensor.data[index] = (float) v;
                    index++;
                }
            }
        }

        if (Use.GPU) {
            int index = 0;
            if (!tensor.TYPE) {
                float[] hostdata = new float[tensor.size];
                for (int d = 0; d < tensor.rows; d++) {
                    double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
                    for (double v : arr) {
                        hostdata[index] = (float) v;
                        index++;
                    }
                }

                cudaMemcpy(tensor.data_gpu, Pointer.to(hostdata), (long) Sizeof.FLOAT * tensor.size, cudaMemcpyHostToDevice);
            }
            else
            {
                short[] hostdata = new short[tensor.size];
                for (int d = 0; d < tensor.rows; d++) {
                    double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Short::parseShort).toArray();
                    for (double v : arr) {
                        hostdata[index] = (short) v;
                        index++;
                    }
                }

                cudaMemcpy(tensor.data_gpu, Pointer.to(hostdata), (long) Sizeof.SHORT * tensor.size, cudaMemcpyHostToDevice);
            }
        }

        return tensor;
    }
    public float[] toArray() {
        float[] data_h = new float[this.rows * this.columns * this.depth];
        JCublas2.cublasGetVector(data_h.length, Sizeof.FLOAT, data_gpu, 1, Pointer.to(data_h), 1);
        return data_h;
    }

    public void free() {
        //if (data_gpu != null) JCuda.cudaFree(data_gpu);
        //if (rowsIndex_gpu != null) JCuda.cudaFree(rowsIndex_gpu);
        //if (columnsIndex_gpu != null) JCuda.cudaFree(columnsIndex_gpu);
    }
}
