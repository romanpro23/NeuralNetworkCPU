package nnarrays;

import lombok.Getter;
import utilities.Use;

import java.io.FileWriter;
import java.io.IOException;
import java.lang.ref.WeakReference;
import java.util.Arrays;
import java.util.Scanner;

import static utilities.GPUInit.allocated;
import static utilities.GPUInit.allocatedUse;

public class NNTensor4D extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;
    @Getter
    private final int length;

    @Getter
    private int[] depthIndex, lengthIndex, rowIndex;

    public NNTensor4D(int depth, int length, int row, int column) {
        super(depth * length * row * column);
        this.column = column;
        this.row = row;
        this.depth = depth;
        this.length = length;

        countAxes = 4;
        initialize();
    }

    private void initialize() {
        depthIndex = new int[depth];
        lengthIndex = new int[length];
        rowIndex = new int[row];

        if (Use.CPU) {
            int sq = column * row * length;
            int sql = column * row;
            for (int i = 0; i < depth; i++) {
                depthIndex[i] = i * sq;
            }
            for (int i = 0; i < length; i++) {
                lengthIndex[i] = i * sql;
            }
            for (int i = 0; i < row; i++) {
                rowIndex[i] = i * column;
            }
        }
    }

    @Override
    public int size() {
        return length * depth * row * column;
    }

    public int depth() {
        return depth;
    }

    public int length() {
        return length;
    }

    public int row() {
        return row;
    }

    public int column() {
        return column;
    }

    public int[] depthIndex() {
        return depthIndex;
    }

    public int[] rowIndex() {
        return rowIndex;
    }

    public int[] lengthIndex() {
        return lengthIndex;
    }

    @Override
    public int[] shape() {
        return new int[]{depth, length, row, column};
    }

    public float get(int i, int j, int k, int l) {
        return data[depthIndex[i] + lengthIndex[j] + rowIndex[k] + l];
    }

    public void convolution(NNTensor input, NNTensor error, int step, int padY, int padX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;

        for (int y = -padY, h = 0; h < error.getRows(); y += step, h++) {
            for (int x = -padX, w = 0; w < error.getColumns(); x += step, w++) {
                outputIndex = error.getRowsIndex()[h] + error.getColumnsIndex()[w];
                for (int d = 0; d < depth; d++, outputIndex++) {
                    for (int j = 0; j < length; j++) {
                        y0 = y + j;
                        if (y0 < 0 || y0 >= input.getRows()) {
                            continue;
                        }
                        w0 = depthIndex[d] + lengthIndex[j];
                        for (int k = 0; k < row; k++) {
                            x0 = x + k;
                            if (x0 < 0 || x0 >= input.getColumns()) {
                                continue;
                            }
                            inputIndex = input.getRowsIndex()[y0] + input.getColumnsIndex()[x0];
                            weightIndex = w0 + rowIndex[k];
                            for (int c = 0; c < column; c++, inputIndex++, weightIndex++) {
                                data[weightIndex] += input.data[inputIndex] * error.data[outputIndex];
                            }
                        }
                    }
                }
            }
        }
    }

    public void
    addMatrixDot(NNMatrix input, NNTensor error) {
        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < length; j++) {
                for (int l = 0; l < row; l++) {
                    for (int k = 0; k < column; k++) {
                        add(i, j, l, k, input.get(j, k) * error.get(i, j, l));
                    }
                }
            }
        }
    }

    public NNMatrix dot(NNTensor error) {
        NNMatrix result = new NNMatrix(length, column);

        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < length; j++) {
                for (int k = 0; k < column; k++) {
                    for (int l = 0; l < row; l++) {
                        result.add(j, k, error.get(i, j, l) * get(i, j, l, k));
                    }
                }
            }
        }
        return result;
    }

    public void dilatedConvolution(NNTensor input, NNTensor error, int step, int padY, int padX, int dilatationY, int dilatationX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;

        int dilY = (((length() - 1) * dilatationY + 1) - length()) / 2;
        int dilX = (((row() - 1) * dilatationX + 1) - row()) / 2;

        for (int y = -padY - dilY, h = 0; h < error.getRows(); y += step, h++) {
            for (int x = -padX - dilX, w = 0; w < error.getColumns(); x += step, w++) {
                outputIndex = error.getRowsIndex()[h] + error.getColumnsIndex()[w];
                for (int d = 0; d < depth; d++, outputIndex++) {
                    for (int j = 0; j < length; j++) {
                        y0 = y + j * dilatationY;
                        if (y0 < 0 || y0 >= input.getRows()) {
                            continue;
                        }
                        w0 = depthIndex[d] + lengthIndex[j];
                        for (int k = 0; k < row; k++) {
                            x0 = x + k * dilatationX;
                            if (x0 < 0 || x0 >= input.getColumns()) {
                                continue;
                            }
                            inputIndex = input.getRowsIndex()[y0] + input.getColumnsIndex()[x0];
                            weightIndex = w0 + rowIndex[k];
                            for (int c = 0; c < column; c++, inputIndex++, weightIndex++) {
                                data[weightIndex] += input.data[inputIndex] * error.data[outputIndex];
                            }
                        }
                    }
                }
            }
        }
    }

    public void groupConvolution(NNTensor input, NNTensor error, int step, int padY, int padX, int countGroup) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        int sizeGroupKernel = depth / countGroup;

        for (int g = 0, gI, gO; g < countGroup; g++) {
            gI = g * column;
            gO = g * sizeGroupKernel;
            for (int y = -padY, h = 0; h < error.getRows(); y += step, h++) {
                for (int x = -padX, w = 0; w < error.getColumns(); x += step, w++) {
                    outputIndex = error.getRowsIndex()[h] + error.getColumnsIndex()[w] + gO;
                    for (int d = 0; d < sizeGroupKernel; d++, outputIndex++) {
                        for (int j = 0; j < length; j++) {
                            y0 = y + j;
                            if (y0 < 0 || y0 >= input.getRows()) {
                                continue;
                            }
                            w0 = depthIndex[d + gO] + lengthIndex[j];
                            for (int k = 0; k < row; k++) {
                                x0 = x + k;
                                if (x0 < 0 || x0 >= input.getColumns()) {
                                    continue;
                                }
                                inputIndex = input.getRowsIndex()[y0] + input.getColumnsIndex()[x0] + gI;
                                weightIndex = w0 + rowIndex[k];
                                for (int c = 0; c < column; c++, weightIndex++, inputIndex++) {
                                    data[weightIndex] += input.data[inputIndex] * error.data[outputIndex];
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    public void add(int i, int j, int k, int l, float val) {
        data[depthIndex[i] + lengthIndex[j] + rowIndex[k] + l] += val;
    }

    public void set(int i, int j, int k, int l, float val) {
        data[depthIndex[i] + lengthIndex[j] + rowIndex[k] + l] = val;
    }

    public void convolutionTranspose(NNTensor input, NNTensor error, int paddingY, int paddingX) {
        int x0, y0, inputIndex, weightIndex, w0, outputIndex;
        int padY = length - 1 - paddingY;
        int padX = row - 1 - paddingX;
        int wCore = row - 1;
        int hCore = length - 1;
        int hC, wC;

        for (int y = -padY, h = 0; h < error.getRows(); y++, h++) {
            for (int x = -padX, w = 0; w < error.getColumns(); x++, w++) {
                outputIndex = error.getRowsIndex()[h] + error.getColumnsIndex()[w];
                for (int d = 0; d < column; d++, outputIndex++) {
                    for (int j = 0; j < length; j++) {
                        y0 = y + j;
                        if (y0 < 0 || y0 >= input.getRows()) {
                            continue;
                        }
                        hC = hCore - j;
                        w0 = d + lengthIndex[hC];
                        for (int k = 0; k < row; k++) {
                            x0 = x + k;
                            if (x0 < 0 || x0 >= input.getColumns()) {
                                continue;
                            }
                            inputIndex = input.getRowsIndex()[y0] + input.getColumnsIndex()[x0];
                            wC = wCore - k;
                            weightIndex = w0 + rowIndex[wC];

                            for (int c = 0; c < depth; c++, inputIndex++) {
                                data[depthIndex[c] + weightIndex] += input.data[inputIndex] * error.data[outputIndex];
                            }
                        }
                    }
                }
            }
        }
    }

    public void save(FileWriter writer) throws IOException {
        writer.write(depth + " " + length + " " + row + " " + column + "\n");
        for (int d = 0; d < depth; d++) {
            for (int l = 0; l < length; l++) {
                for (int i = 0; i < row; i++) {
                    for (int j = 0; j < column; j++) {
                        writer.write(get(d, l, i, j) + " ");
                    }
                }
                writer.write("\n");
                writer.flush();
            }
            writer.write("\n");
        }
        writer.flush();
    }

    public static NNTensor4D read(Scanner scanner) {
        int[] size = Arrays.stream(scanner.nextLine().split(" ")).mapToInt(Integer::parseInt).toArray();
        NNTensor4D tensor = new NNTensor4D(size[0], size[1], size[2], size[3]);
        int index = 0;
        for (int d = 0; d < tensor.depth; d++) {
            for (int l = 0; l < tensor.length; l++) {
                double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
                for (double v : arr) {
                    tensor.data[index] = (float) v;
                    index++;
                }
            }
            scanner.nextLine();
        }

        return tensor;
    }
}
