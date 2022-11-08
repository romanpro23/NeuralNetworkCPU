package nnarrays;

import lombok.Getter;
import lombok.SneakyThrows;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

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

    public NNTensor(int[] size) {
        this(size[0], size[1], size[2]);
    }

    private void initialize() {
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

    public NNTensor(int rows, int columns, int depth, float[] data) {
        super(data);
        this.depth = depth;
        this.columns = columns;
        this.rows = rows;
        countAxes = 3;

        initialize();
    }

    @Override
    public int[] getSize() {
        return new int[]{rows, columns, depth};
    }

    public float get(int i, int j, int k) {
        return data[rowsIndex[i] + columnsIndex[j] + k];
    }

    public void set(int i, int j, int k, float value) {
        data[rowsIndex[i] + columnsIndex[j] + k] = value;
    }

    public void shuffle(NNTensor input, int countGroup){
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

    public void backShuffle(NNTensor input, int countGroup){
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
                        data[index] = error.data[k];
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
                    data[index] = error.data[k];
                }
            }
        }
    }

    public void convolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        float val;

        for (int y = -padY, h = 0; h < rows; y += step, h++) {
            for (int x = -padX, w = 0; w < columns; x += step, w++) {
                outputIndex = rowsIndex[h] + columnsIndex[w];
                for (int d = 0; d < weight.depth(); d++, outputIndex++) {
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

    public void dilatedConvolution(NNTensor input, NNTensor4D weight, int step, int padY, int padX, int dilatationY, int dilatationX) {
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

    public void transposeConvolution(NNTensor input, NNTensor4D weight, int paddingY, int paddingX) {
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

    public void transposeDilatedConvolution(NNTensor input, NNTensor4D weight, int paddingY, int paddingX, int dilatationY, int dilatationX) {
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

    public void transposeGroupConvolution(NNTensor input, NNTensor4D weight, int paddingY, int paddingX, int countGroup) {
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

    public NNTensor stride(int stride) {
        if (stride == 1) {
            return this;
        }
        NNTensor result = new NNTensor(rows * stride, columns * stride, depth);
        int inputIndex, outpuIndex;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                inputIndex = rowsIndex[i] + columnsIndex[j];
                outpuIndex = result.rowsIndex[i * stride] + result.columnsIndex[j * stride];
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
        writer.write(rows + " " + columns + " " + depth + "\n");
        for (int d = 0; d < rows; d++) {
            for (int i = 0; i < columns; i++) {
                for (int j = 0; j < depth; j++) {
                    writer.write(get(d, i, j) + " ");
                }
            }
            writer.write("\n");
            writer.flush();
        }
    }

    public static NNTensor read(Scanner scanner) {
        int[] size = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        NNTensor tensor = new NNTensor(size[0], size[1], size[2]);
        int index = 0;
        for (int d = 0; d < tensor.rows; d++) {
            double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
            for (double v : arr) {
                tensor.data[index] = (float) v;
                index++;
            }
        }

        return tensor;
    }
}
