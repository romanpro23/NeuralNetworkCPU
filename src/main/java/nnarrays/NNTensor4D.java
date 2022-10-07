package nnarrays;

import lombok.Getter;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NNTensor4D extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int depth;
    @Getter
    private final int length;

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
    public int[] getSize() {
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
