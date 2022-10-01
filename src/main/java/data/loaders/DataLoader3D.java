package data.loaders;

import data.network_train.NNData3D;
import nnarrays.NNTensor;
import nnarrays.NNVector;

import java.util.ArrayList;
import java.util.Collections;

public abstract class DataLoader3D extends DataLoader {
    protected ArrayList<ImageData3D> train;
    protected ArrayList<ImageData3D> test;

    private int curTrain = 0, curTest = 0;

    @Override
    public NNData3D getNextTrainData(int sizeBatch) {
        int size = sizeBatch;
        if (curTrain + sizeBatch >= train.size()) {
            size = train.size() - curTrain;
        }

        NNTensor[] input = new NNTensor[size];
        NNVector[] output = new NNVector[size];

        for (int i = 0; i < size; i++) {
            input[i] = train.get(curTrain).getInputs();
            output[i] = train.get(curTrain).getOutputs();

            curTrain++;
        }

        if(size != sizeBatch || curTrain == train.size()){
            curTrain = 0;
        }

        return new NNData3D(input, output);
    }

    @Override
    public NNData3D getNextTestData(int sizeBatch) {
        int size = sizeBatch;
        if (curTest + sizeBatch >= test.size()) {
            size = test.size() - curTest;
        }

        NNTensor[] input = new NNTensor[size];
        NNVector[] output = new NNVector[size];

        for (int i = 0; i < size; i++) {
            input[i] = test.get(curTest).getInputs();
            output[i] = test.get(curTest).getOutputs();

            curTest++;
        }

        if(size != sizeBatch || curTest == test.size()){
            curTest = 0;
            Collections.shuffle(test);
        }

        return new NNData3D(input, output);
    }
}
