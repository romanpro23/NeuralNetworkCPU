package data.loaders;

import data.loaders.DataLoader;
import data.network_train.NNData1D;
import nnarrays.NNVector;

import java.util.ArrayList;
import java.util.Collections;

public abstract class DataLoader1D extends DataLoader {
    protected ArrayList<ImageData1D> train;
    protected ArrayList<ImageData1D> test;

    private int curTrain = 0, curTest = 0;

    @Override
    public NNData1D getNextTrainData(int sizeBatch) {
        int size = sizeBatch;
        if (curTrain + sizeBatch >= train.size()) {
            size = train.size() - curTrain;
        }

        NNVector[] input = new NNVector[size];
        NNVector[] output = new NNVector[size];

        for (int i = 0; i < size; i++) {
            input[i] = train.get(curTrain).getInputs();
            output[i] = train.get(curTrain).getOutputs();

            curTrain++;
        }

        if(size != sizeBatch || curTrain == train.size()){
            curTrain = 0;
            //Collections.shuffle(train);
        }

        return new NNData1D(input, output);
    }

    @Override
    public NNData1D getNextTestData(int sizeBatch) {
        int size = sizeBatch;
        if (curTest + sizeBatch >= test.size()) {
            size = test.size() - curTest;
        }

        NNVector[] input = new NNVector[size];
        NNVector[] output = new NNVector[size];

        for (int i = 0; i < size; i++) {
            input[i] = test.get(curTest).getInputs();
            output[i] = test.get(curTest).getOutputs();

            curTest++;
        }

        if(size != sizeBatch || curTest == test.size()){
            curTest = 0;
            //Collections.shuffle(test);
        }

        return new NNData1D(input, output);
    }
}
