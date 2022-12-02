package nnarrays;

import lombok.SneakyThrows;

import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import java.util.Scanner;

public class NNVector extends NNArray {
    public NNVector(int length) {
        super(length);
        countAxes = 1;
    }

    public NNVector(NNVector vector) {
        super(vector.size);
        countAxes = 1;
    }

    public NNVector(NNArray array) {
        super(array.data);
        countAxes = 1;
    }

    public NNVector(float[] data) {
        super(data);
        countAxes = 1;
    }

    public NNVector(int[] data) {
        super(data);
        countAxes = 1;
    }

    public NNVector dot(NNMatrix matrix) {
        NNVector result = new NNVector(matrix.getRow());

        for (int i = 0, index = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++, index++) {
                result.data[i] += data[j] * matrix.data[index];
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

        for (int i = 0, index = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++, index++) {
                result.data[j] += data[i] * matrix.data[index];
            }
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
        for (int i = 0; i < size; i++) {
            data[i] += vector.data[i];
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
        int index = 0;
        for (int i = 0; i < matrix.getRow(); i++) {
            for (int k = 0; k < matrix.getColumn(); k++, index++) {
                data[k] += matrix.data[index];
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
        writer.write(size + "\n");
        for (int i = 0; i < size; i++) {
            writer.write(data[i] + " ");
            if (i % 1000 == 0) {
                writer.flush();
            }
        }
        writer.write("\n");
        writer.flush();
    }

    public static NNVector read(Scanner scanner) {
        NNVector vector = new NNVector(Integer.parseInt(scanner.nextLine()));
        double[] arr = Arrays.stream(scanner.nextLine().split(" ")).mapToDouble(Float::parseFloat).toArray();
        for (int j = 0; j < vector.size; j++) {
            vector.data[j] = (float) arr[j];
        }
        return vector;
    }
}
