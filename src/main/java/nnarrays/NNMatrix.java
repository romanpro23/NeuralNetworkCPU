package nnarrays;

import lombok.Getter;
import lombok.SneakyThrows;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NNMatrix extends NNArray{
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
    }

    public NNMatrix(NNMatrix matrix) {
        this(matrix.row, matrix.column);
    }

    public float get(int i, int j){
        return data[rowIndex[i] + j];
    }

    @SneakyThrows
    public void add(NNMatrix matrix){
        if(size != matrix.size){
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] += matrix.data[i];
        }
    }

    public void add(int i, int j, float val){
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
}
