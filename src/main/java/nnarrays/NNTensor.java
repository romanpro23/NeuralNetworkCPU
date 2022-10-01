package nnarrays;

import lombok.Getter;
import lombok.SneakyThrows;

import java.util.Arrays;

public class NNTensor extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;
    @Getter
    private int[] depthIndex;
    @Getter
    private int[] rowIndex;

    public NNTensor(int depth, int row, int column) {
        super(column * row * depth);
        this.column = column;
        this.row = row;
        this.depth = depth;
        countAxes = 3;

        initialize();
    }

    private void initialize() {
        depthIndex = new int[depth];
        rowIndex = new int[row];
        int sq = column * row;
        for (int i = 0; i < depth; i++) {
            depthIndex[i] = i * sq;
        }
        for (int i = 0; i < row; i++) {
            rowIndex[i] = i * column;
        }
    }

    public NNTensor(int depth, int row, int column, float[] data) {
        super(data);
        this.column = column;
        this.row = row;
        this.depth = depth;
        countAxes = 3;

        initialize();
    }

    @Override
    public int[] getSize() {
        return new int[]{depth, row, column};
    }

    public float get(int i, int j, int k) {
        return data[depthIndex[i] + rowIndex[j] + k];
    }

    public void set(int i, int j, int k, float value) {
        data[depthIndex[i] + rowIndex[j] + k] = value;
    }

    public void maxPool(NNTensor input, int heightKernel, int widthKernel, int step, int paddingY, int paddingX) {
        int x0, y0, inputIndex, outputIndex, outIndex;
        Arrays.fill(data, -1000);

        for (int y = -paddingY, h = 0; h < depth; y += step, h++) {
            for (int x = -paddingX, w = 0; w < row; x += step, w++) {
                outputIndex = depthIndex[h] + rowIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.depth) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.row) {
                            continue;
                        }
                        inputIndex = input.depthIndex[y0] + input.rowIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < column; d++, inputIndex++, outIndex++) {
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

        for (int y = 0; y < input.depth; y++) {
            for (int x = 0; x < input.row; x++) {
                inIndex = input.depthIndex[y] + input.rowIndex[x];
                for (int j = 0; j < heightKernel; j++) {
                    yIndex = y * heightKernel + j;
                    for (int k = 0; k < widthKernel; k++) {
                        xIndex = x * widthKernel + k;
                        outputIndex = depthIndex[yIndex] + rowIndex[xIndex];
                        inputIndex = inIndex;
                        for (int d = 0; d < column; d++, outputIndex++, inputIndex++) {
                            data[outputIndex] = input.data[inputIndex];
                        }
                    }
                }
            }
        }
    }

    public void backUpSampling(NNTensor input, int heightKernel, int widthKernel) {
        int xIndex, yIndex, inputIndex, outputIndex, outIndex;

        for (int y = 0; y < depth; y++) {
            for (int x = 0; x < row; x++) {
                outIndex = depthIndex[y] + rowIndex[x];
                for (int j = 0; j < heightKernel; j++) {
                    yIndex = y * heightKernel + j;
                    for (int k = 0; k < widthKernel; k++) {
                        outputIndex = outIndex;
                        xIndex = x * widthKernel + k;
                        inputIndex = input.depthIndex[yIndex] + input.rowIndex[xIndex];
                        for (int d = 0; d < column; d++, outputIndex++, inputIndex++) {
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

        for (int y = -paddingY, h = 0; h < depth; y += step, h++) {
            for (int x = -paddingX, w = 0; w < row; x += step, w++) {
                outputIndex = depthIndex[h] + rowIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.depth) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.row) {
                            continue;
                        }
                        inputIndex = input.depthIndex[y0] + input.rowIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < column; d++, inputIndex++, outIndex++) {
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

        for (int y = -paddingY, h = 0; h < outputs.depth; y += step, h++) {
            for (int x = -paddingX, w = 0; w < outputs.row; x += step, w++) {
                outputIndex = outputs.depthIndex[h] + outputs.rowIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.depth) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.row) {
                            continue;
                        }
                        inputIndex = input.depthIndex[y0] + input.rowIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < column; d++, inputIndex++, outIndex++) {
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

        for (int y = -paddingY, h = 0; h < outputs.depth; y += step, h++) {
            for (int x = -paddingX, w = 0; w < outputs.row; x += step, w++) {
                outputIndex = outputs.depthIndex[h] + outputs.rowIndex[w];
                for (int j = 0; j < heightKernel; j++) {
                    y0 = y + j;
                    if (y0 < 0 || y0 >= input.depth) {
                        continue;
                    }
                    for (int k = 0; k < widthKernel; k++) {
                        x0 = x + k;
                        if (x0 < 0 || x0 >= input.row) {
                            continue;
                        }
                        inputIndex = input.depthIndex[y0] + input.rowIndex[x0];
                        outIndex = outputIndex;
                        for (int d = 0; d < input.column; d++, inputIndex++, outIndex++) {
                            this.data[inputIndex] += error.data[outIndex];
                        }
                    }
                }
            }
        }
    }

    public void backGlobalMaxPool(NNTensor input, NNVector output, NNVector error) {
        int index = 0;
        for (int i = 0; i < input.getDepth(); i++) {
            for (int j = 0; j < input.getRow(); j++) {
                for (int k = 0; k < input.getColumn(); k++, index++) {
                    if (output.data[k] == input.data[index]) {
                        data[index] = error.data[k];
                    }
                }
            }
        }
    }

    public void backGlobalAveragePool(NNVector error) {
        int index = 0;
        error.div(depth * row);
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < row; j++) {
                for (int k = 0; k < column; k++, index++) {
                    data[index] = error.data[k];
                }
            }
        }
    }

    public void convolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        float val;

        for (int y = -padY, h = 0; h < depth; y += step, h++) {
            for (int x = -padX, w = 0; w < row; x += step, w++) {
                outputIndex = depthIndex[h] + rowIndex[w];
                for (int d = 0; d < weight.depth(); d++, outputIndex++) {
                    val = 0;
                    for (int j = 0; j < weight.length(); j++) {
                        y0 = y + j;
                        if (y0 < 0 || y0 >= input.depth) {
                            continue;
                        }
                        w0 = weight.depthIndex()[d] + weight.lengthIndex()[j];
                        for (int k = 0; k < weight.row(); k++) {
                            x0 = x + k;
                            if (x0 < 0 || x0 >= input.row) {
                                continue;
                            }
                            inputIndex = input.depthIndex[y0] + input.rowIndex[x0];
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

    public void transposeConvolution(NNTensor input, NNTensor4D weight, int paddingY, int paddingX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        int padY = weight.length() - 1 - paddingY;
        int padX = weight.row() - 1 - paddingX;
        int wCore = weight.row() - 1;
        int hCore = weight.length() - 1;
        int hC, wC;

        float val;

        for (int y = -padY, h = 0; h < depth; y++, h++) {
            for (int x = -padX, w = 0; w < row; x++, w++) {
                outputIndex = depthIndex[h] + rowIndex[w];
                for (int d = 0; d < weight.column(); d++, outputIndex++) {
                    val = 0;
                    for (int j = 0; j < weight.length(); j++) {
                        y0 = y + j;
                        if (y0 < 0 || y0 >= input.depth) {
                            continue;
                        }
                        hC = hCore - j;
                        w0 = weight.lengthIndex()[hC] + d;
                        for (int k = 0; k < weight.row(); k++) {
                            x0 = x + k;
                            if (x0 < 0 || x0 >= input.row) {
                                continue;
                            }
                            inputIndex = input.depthIndex[y0] + input.rowIndex[x0];
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

    public void add(int i, int j, int k, float val) {
        data[depthIndex[i] + rowIndex[j] + k] += val;
    }

    @SneakyThrows
    public void add(NNVector vector) {
        if (column != vector.size) {
            throw new Exception("Array has difference size");
        }
        int inputIndex;
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < row; j++) {
                inputIndex = depthIndex[i] + rowIndex[j];
                for (int k = 0; k < column; k++, inputIndex++) {
                    data[inputIndex] += vector.data[k];
                }
            }
        }
    }

    public NNTensor stride(int stride) {
        if (stride == 1) {
            return this;
        }
        NNTensor result = new NNTensor(depth * stride, row * stride, column);
        int inputIndex, outpuIndex;
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < row; j++) {
                inputIndex = depthIndex[i] + rowIndex[j];
                outpuIndex = result.depthIndex[i * stride] + result.rowIndex[j * stride];
                for (int k = 0; k < column; k++, inputIndex++, outpuIndex++) {
                    result.data[outpuIndex] = data[inputIndex];
                }
            }
        }
        return result;
    }
}
