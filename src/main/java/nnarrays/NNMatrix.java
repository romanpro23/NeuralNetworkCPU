package nnarrays;

import lombok.Getter;
import lombok.SneakyThrows;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NNMatrix extends NNArray {
    @Getter
    private final int column;
    @Getter
    private final int row;
    @Getter
    private final int[] rowIndex;

    public NNMatrix(int row, int column) {
        super(column * row);
        this.column = column;
        this.row = row;

        rowIndex = new int[row];
        for (int i = 0; i < row; i++) {
            rowIndex[i] = i * column;
        }
        countAxes = 2;
    }

    public NNMatrix(int row, int column, float[] data) {
        super(data);
        this.column = column;
        this.row = row;

        rowIndex = new int[row];
        for (int i = 0; i < row; i++) {
            rowIndex[i] = i * column;
        }
        countAxes = 2;
    }

    public NNMatrix(NNMatrix matrix) {
        this(matrix.row, matrix.column);
    }

    public NNVector[] toVectors(){
        NNVector[] vectors = new NNVector[row];
        for (int i = 0; i < row; i++) {
            vectors[i] = new NNVector(column);
            System.arraycopy(data, rowIndex[i], vectors[i].data, 0, column);
        }

        return vectors;
    }

    public void set(NNVector vector, int index_t){
        int index = rowIndex[index_t];
        System.arraycopy(vector.data, 0, data, index, vector.size);
    }

    @Override
    public int[] getSize() {
        return new int[]{row, column};
    }

    public float get(int i, int j) {
        return data[rowIndex[i] + j];
    }

    @SneakyThrows
    public void add(NNMatrix matrix) {
        if (size != matrix.size) {
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] += matrix.data[i];
        }
    }

    public void add(int i, int j, float val) {
        data[rowIndex[i] + j] += val;
    }

    public NNMatrix transpose() {
        NNMatrix nnMatrix = new NNMatrix(this.column, this.row);
        int index;
        for (int i = 0; i < row; i++) {
            index = rowIndex[i];
            for (int j = 0; j < column; j++, index++) {
                nnMatrix.data[i + nnMatrix.rowIndex[j]] = data[index];
            }
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
        int inputIndex;
        for (int i = 0; i < row; i++) {
            inputIndex = rowIndex[i];
            for (int k = 0; k < column; k++, inputIndex++) {
                data[inputIndex] += vector.data[k];
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
}
