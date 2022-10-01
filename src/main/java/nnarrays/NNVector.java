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

    public NNVector mul(NNMatrix matrix) {
        NNVector result = new NNVector(matrix.getRow());

        for (int i = 0, index = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++, index++) {
                result.data[i] += data[j] * matrix.data[index];
            }
        }

        return result;
    }

    public void momentumAverage(NNArray array, final float decay) {
        for (int i = 0; i < size; i++) {
            data[i] += (array.data[i] - data[i]) * decay;
        }
    }

    public NNVector mulT(NNMatrix matrix) {
        NNVector result = new NNVector(matrix.getColumn());

        for (int i = 0, index = 0; i < matrix.getRow(); i++) {
            for (int j = 0; j < matrix.getColumn(); j++, index++) {
                result.data[j] += data[i] * matrix.data[index];
            }
        }

        return result;
    }

    public void globalMaxPool(NNTensor input) {
        int index = 0;
        fill(-1000);
        for (int i = 0; i < input.getDepth(); i++) {
            for (int j = 0; j < input.getRow(); j++) {
                for (int k = 0; k < size; k++, index++) {
                    if (data[k] < input.data[index]) {
                        data[k] = input.data[index];
                    }
                }
            }
        }
    }

    public void globalAveragePool(NNTensor input) {
        int index = 0;
        for (int i = 0; i < input.getDepth(); i++) {
            for (int j = 0; j < input.getRow(); j++) {
                for (int k = 0; k < size; k++, index++) {
                    data[k] += input.data[index];
                }
            }
        }
        div(input.getDepth() * input.getRow());
    }

    public NNMatrix mulVector(NNVector vector) {
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
    public void addProduct(NNVector vector1, NNVector vector2) {
        if (size != vector1.size || size != vector2.size) {
            System.out.println(vector1.size + " " + vector2.size);
            throw new Exception("Vector has difference size");
        }
        for (int i = 0; i < size; i++) {
            data[i] += vector1.data[i] * vector2.data[i];
        }
    }

    @SneakyThrows
    public void add(NNTensor tensor) {
        if (size != tensor.getColumn()) {
            throw new Exception("Vector has difference size");
        }
        int index = 0;
        for (int i = 0; i < tensor.getDepth(); i++) {
            for (int j = 0; j < tensor.getRow(); j++) {
                for (int k = 0; k < tensor.getColumn(); k++, index++) {
                    data[k] += tensor.data[index];
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
