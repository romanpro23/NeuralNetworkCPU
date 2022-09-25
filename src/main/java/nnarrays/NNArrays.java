package nnarrays;

import lombok.SneakyThrows;

public final class NNArrays {
    public static NNVector[] isVector(NNArray[] batch) {
        return (NNVector[]) batch;
    }

    public static NNVector[] toVector(NNArray[] batch) {
        NNVector[] arr = new NNVector[batch.length];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = new NNVector(batch[i]);
        }

        return arr;
    }

    public static NNArray[] concat(NNArray[] first, NNArray[] second) {
        if (first[0].countAxes == 1) {
            return concatVector(first, second);
        } else if (first[0].countAxes == 2) {
            return concatMatrix(first, second);
        } else if (first[0].countAxes == 3) {
            return concatTensor(first, second);
        }
        return null;
    }

    public static NNArray[] subArray(NNArray[] first, NNArray[] second) {
        if (first[0].countAxes == 1) {
            return subVector(first, second);
        } else if (first[0].countAxes == 2) {
            return subMatrix(first, second);
        } else if (first[0].countAxes == 3) {
            return subTensor(first, second);
        }
        return null;
    }

    @SneakyThrows
    public static NNVector[] concatVector(NNArray[] first, NNArray[] second) {
        if (first.length != second.length) {
            throw new Exception("Vector has difference size");
        }
        NNVector[] result = new NNVector[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[first[i].size + second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, first[i].size);
            System.arraycopy(second[i].data, 0, data, first[i].size, second[i].size);
            result[i] = new NNVector(data);
        }
        return result;
    }

    @SneakyThrows
    public static NNVector[] subVector(NNArray[] first, NNArray[] second) {
        NNVector[] result = new NNVector[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, second[i].size);
            result[i] = new NNVector(data);
        }
        return result;
    }

    @SneakyThrows
    public static NNMatrix[] subMatrix(NNArray[] first, NNArray[] second) {
        NNMatrix[] result = new NNMatrix[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, second[i].size);
            int row = second[i].getSize()[0];
            int column = second[i].getSize()[1];
            result[i] = new NNMatrix(row, column, data);
        }
        return result;
    }

    @SneakyThrows
    public static NNMatrix[] concatMatrix(NNArray[] first, NNArray[] second) {
        if (first.length != second.length) {
            throw new Exception("Vector has difference size");
        }
        NNMatrix[] result = new NNMatrix[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[first[i].size + second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, first[i].size);
            System.arraycopy(second[i].data, 0, data, first[i].size, second[i].size);
            int row = first[i].getSize()[0] + second[i].getSize()[0];
            int column = first[i].getSize()[1];
            result[i] = new NNMatrix(row, column, data);
        }
        return result;
    }

    @SneakyThrows
    public static NNTensor[] subTensor(NNArray[] first, NNArray[] second) {
        NNTensor[] result = new NNTensor[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, second[i].size);
            int depth = second[i].getSize()[0];
            int row = second[i].getSize()[1];
            int column = second[i].getSize()[2];
            result[i] = new NNTensor(depth, row, column, data);
        }
        return result;
    }

    @SneakyThrows
    public static NNTensor[] concatTensor(NNArray[] first, NNArray[] second) {
        if (first.length != second.length) {
            throw new Exception("Vector has difference size");
        }
        NNTensor[] result = new NNTensor[first.length];
        for (int i = 0; i < first.length; i++) {
            float[] data = new float[first[i].size + second[i].size];

            System.arraycopy(first[i].data, 0, data, 0, first[i].size);
            System.arraycopy(second[i].data, 0, data, first[i].size, second[i].size);
            int depth = first[i].getSize()[0] + second[i].getSize()[0];
            int row = first[i].getSize()[1];
            int column = first[i].getSize()[2];
            result[i] = new NNTensor(depth, row, column, data);
        }
        return result;
    }

    public static NNMatrix mul(NNVector[] first, NNVector[] second) {
        NNMatrix result = new NNMatrix(first[0].size, second[0].size);

        for (int i = 0; i < first.length; i++) {
            for (int j = 0, index = 0; j < first[i].size(); j++) {
                for (int k = 0; k < second[i].size(); k++, index++) {
                    result.data[index] += first[i].data[j] * second[i].data[k];
                }
            }
        }

        return result;
    }

    public static NNMatrix[] isMatrix(NNArray[] batch) {
        return (NNMatrix[]) batch;
    }

    public static NNTensor[] isTensor(NNArray[] batch) {
        return (NNTensor[]) batch;
    }

    public static float sum(NNArray array) {
        float sum = 0;
        for (int i = 0; i < array.size; i++) {
            sum += array.data[i];
        }

        return sum;
    }

    @SneakyThrows
    public static NNArray sub(NNArray first, NNArray second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = first.data[i] - second.data[i];
        }

        return result;
    }

    @SneakyThrows
    public static NNVector div(NNVector first, NNVector second) {
        if (first.size != second.size) {
            throw new Exception("Vector has difference size");
        }
        NNVector result = new NNVector(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = first.data[i] / second.data[i];
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derBinaryCrossEntropy(NNArray outputs, NNArray idealOutputs) {
        if (outputs.size != idealOutputs.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (outputs.data[i] - idealOutputs.data[i]) / ((1 - outputs.data[i]) * outputs.data[i] + 0.00000001f);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derCrossEntropy(NNArray outputs, NNArray idealOutputs) {
        if (outputs.size != idealOutputs.size) {
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = -idealOutputs.data[i] / (outputs.data[i] + 0.00000001f);
        }

        return result;
    }
}

