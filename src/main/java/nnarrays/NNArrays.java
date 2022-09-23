package nnarrays;

import lombok.SneakyThrows;

public final class NNArrays {
    public static NNVector[] isVector(NNArray[] batch){
        return (NNVector[]) batch;
    }

    public static NNVector[] toVector(NNArray[] batch){
        NNVector[] arr = new NNVector[batch.length];
        for (int i = 0; i < arr.length; i++) {
            arr[i] = new NNVector(batch[i]);
        }

        return arr;
    }

    public static NNMatrix mul(NNVector[] first, NNVector[] second){
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

    public static NNMatrix[] isMatrix(NNArray[] batch){
        return (NNMatrix[]) batch;
    }

    public static float sum(NNArray array){
        float sum = 0;
        for (int i = 0; i < array.size; i++) {
            sum += array.data[i];
        }

        return sum;
    }

    @SneakyThrows
    public static NNArray sub(NNArray first, NNArray second){
        if(first.size != second.size){
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = first.data[i] - second.data[i];
        }

        return result;
    }

    @SneakyThrows
    public static NNVector div(NNVector first, NNVector second){
        if(first.size != second.size){
            throw new Exception("Vector has difference size");
        }
        NNVector result = new NNVector(first.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = first.data[i] / second.data[i];
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derBinaryCrossEntropy(NNArray outputs, NNArray idealOutputs){
        if(outputs.size != idealOutputs.size){
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] = (outputs.data[i] - idealOutputs.data[i]) / ((1 - outputs.data[i]) * outputs.data[i] + 0.00000001f);
        }

        return result;
    }

    @SneakyThrows
    public static NNArray derCrossEntropy(NNArray outputs, NNArray idealOutputs){
        if(outputs.size != idealOutputs.size){
            throw new Exception("Vector has difference size");
        }
        NNArray result = new NNArray(outputs.size);
        for (int i = 0; i < result.size; i++) {
            result.data[i] =  -idealOutputs.data[i] / (outputs.data[i] + 0.00000001f);
        }

        return result;
    }
}

